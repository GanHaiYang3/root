import shutil
import glob
from datetime import date, timedelta
import random
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("feature_size", 225302, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 39, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")# 0520 and 0521 (32)  0522(128)
tf.app.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 1000, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")  # 0520(0.0001)  0521(0.0001)
tf.app.flags.DEFINE_float("l2_reg", 0.01, "L2 regularization") # 0520(0.001)  0521(0.01)
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '64,128,256,512,1024', "deep layers")  # 0520 and 0521 ('256,128,64')  0522(512,256,128)
tf.app.flags.DEFINE_string("dropout", '0.90,0.90,0.80,0.8,0.8', "dropout rate") ## 0520(0.9 0.9 0.9)  0521(0.7 0.7 0.5)
tf.app.flags.DEFINE_boolean("batch_norm", True, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")

tf.app.flags.DEFINE_string("nfmdeep_layers", '64,128', "deep layers")  # 0520 and 0521 ('256,128,64')  0522(512,256,128)
tf.app.flags.DEFINE_string("nfmdropout", '0.5,0.8', "dropout rate") ## 0520(0.9 0.9 0.9)  0521(0.7 0.7 0.5)
tf.app.flags.DEFINE_float("nfml2_reg", 0.01, "L2 regularization") # 0520(0.001) 0521(0.01)


tf.app.flags.DEFINE_string("data_dir", '../Data', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", 'model3', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)
    def decode_libsvm(line):
        columns = tf.string_split([line],' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values,splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000) # multi-thread pre-process then prefetch

    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def model_fn(features, labels, mode, params):
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    layers = list(map(int, params["deep_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))

    nfml2_reg = params["nfml2_reg"]
    nfmlayers = list (map (int, params["nfmdeep_layers"].split (',')))
    nfmdropout = list (map (float, params["nfmdropout"].split (',')))

    #------bulid weights------
    FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
    FM_W = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    FM_V = tf.get_variable(name='fm_v', shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())

    feat_ids  = features['feat_ids']
    feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

    with tf.variable_scope("First-order"):
        feat_wgts = tf.nn.embedding_lookup(FM_W, feat_ids)
        y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals),1)

    with tf.variable_scope("Second-order"):
        embeddings = tf.nn.embedding_lookup(FM_V, feat_ids)
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals)
        sum_square = tf.square(tf.reduce_sum(embeddings,1))
        square_sum = tf.reduce_sum(tf.square(embeddings),1)
        y_v = 0.5*tf.reduce_sum(tf.subtract(sum_square, square_sum),1)
        nfmy_v = 0.5 * tf.subtract(sum_square, square_sum)


    with tf.variable_scope ("nfmDeep-part"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_phase = True
        else:
            train_phase = False

        if mode == tf.estimator.ModeKeys.TRAIN:
            nfmy_v = tf.nn.dropout (nfmy_v, keep_prob=nfmdropout[0])  # None * K
        for i in range (len(nfmlayers)):
            nfmy_v = tf.contrib.layers.fully_connected (inputs=nfmy_v, num_outputs=nfmlayers[i],weights_regularizer=tf.contrib.layers.l2_regularizer (nfml2_reg), scope='mlp%d' % i)

            if FLAGS.batch_norm:
                nfmy_v = batch_norm_layer (nfmy_v, train_phase=train_phase,scope_bn='bn_%d' % i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                nfmy_v = tf.nn.dropout (nfmy_v, keep_prob=nfmdropout[i])

        y_deep = tf.contrib.layers.fully_connected (inputs=nfmy_v, num_outputs=1, activation_fn=tf.identity,weights_regularizer=tf.contrib.layers.l2_regularizer (nfml2_reg),scope='deep_out')
        nfmy_d = tf.reshape (y_deep, shape=[-1])

    with tf.variable_scope("Deep-part"):
        if FLAGS.batch_norm:
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
            else:
                train_phase = False
        else:
            normalizer_fn = None
            normalizer_params = None

        deep_inputs = tf.reshape(embeddings,shape=[-1,field_size*embedding_size]) # None * (F*K)
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp%d' % i)
            if FLAGS.batch_norm:
                deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='deep_out')
        y_d = tf.reshape(y_deep,shape=[-1])


    with tf.variable_scope("DeepFM-out"):
        y_bias = FM_B * tf.ones_like(y_d, dtype=tf.float32)
        y = y_bias + y_w + y_v + y_d+nfmy_d
        pred = tf.sigmoid(y)

    predictions={"prob": pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
        l2_reg * tf.nn.l2_loss(FM_W) + \
        l2_reg * tf.nn.l2_loss(FM_V)

    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)

def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z

def main(_):
    tr_files = glob.glob("%s/tr*libsvm" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/va.libsvm" % FLAGS.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob("%s/te.libsvm" % FLAGS.data_dir)
    print("te_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "batch_norm_decay": FLAGS.batch_norm_decay,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout,

        "nfmdropout": FLAGS.nfmdropout,
        "nfml2_reg": FLAGS.nfml2_reg,
        "nfmdeep_layers": FLAGS.nfmdeep_layers,
    }
    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
            log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    DeepFM = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size),max_steps=135000)#135000
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(DeepFM, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        DeepFM.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'infer':
        preds = DeepFM.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="prob")
        ids=[]
        with open(te_files[0], "r") as fr:
            for line in fr:
                features = line.rstrip('\n').split(' ')
                ids.append(features[0])
        with open(FLAGS.data_dir+"/pred.csv", "w") as fo:
            fo.write ("id,click\n")
            for i,prob in enumerate(preds):
                fo.write("%s,%f\n" % (str(ids[i]),prob['prob']))
    elif FLAGS.task_type == 'export':
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        DeepFM.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)

if __name__ == "__main__":
    #------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('batch_norm_decay ', FLAGS.batch_norm_decay)
    print('batch_norm ', FLAGS.batch_norm)
    print('l2_reg ', FLAGS.l2_reg)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
