# -*- coding:utf-8 -*- 
# tensorflow 1.3
## self-define model named "deep and cross network" #

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil

import sys
import tensorflow as tf

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_dir', type=str, default='/tmp/census_model',
    help='Base directory for the model.')
parser.add_argument(
    '--model_type', type=str, default='deep_cross',
    help="Valid model types: {'wide', 'deep', 'wide_deep', 'deep_cross'}.")
parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')
parser.add_argument(
    '--train_data', type=str, default='/tmp/census_data/adult.data',
    help='Path to the training data.')
parser.add_argument(
    '--test_data', type=str, default='/tmp/census_data/adult.test',
    help='Path to the test data.')

def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  age = tf.feature_column.numeric_column('age')
  education_num = tf.feature_column.numeric_column('education_num')
  capital_gain = tf.feature_column.numeric_column('capital_gain')
  capital_loss = tf.feature_column.numeric_column('capital_loss')
  hours_per_week = tf.feature_column.numeric_column('hours_per_week')
  education = tf.feature_column.categorical_column_with_vocabulary_list(
          'education', [
          'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
          'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
          '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
  marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
          'marital_status', [
          'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
          'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
  relationship = tf.feature_column.categorical_column_with_vocabulary_list(
          'relationship', [
          'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
          'Other-relative'])
  workclass = tf.feature_column.categorical_column_with_vocabulary_file(
       key  = 'workclass',
       vocabulary_file  = './data/workclass_vocabulary.mult',
       vocabulary_size  = 10)
  occupation = tf.feature_column.categorical_column_with_vocabulary_file(
       key   = 'occupation', 
       vocabulary_file  = './data/occupation_vocabulary',
       vocabulary_size  = 15)
  age_buckets = tf.feature_column.bucketized_column(
       age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
  columns = [
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
      tf.feature_column.indicator_column(workclass),
      tf.feature_column.indicator_column(education),
      tf.feature_column.indicator_column(marital_status),
      tf.feature_column.indicator_column(relationship),
      tf.feature_column.indicator_column(occupation)]
  return columns
def cross_variable_creat(column_num):
  #print ('in cross_variable_creat, column_num:', column_num, 'type:', type(column_num))
  w = tf.Variable(tf.random_normal((column_num, 1), mean=0.0, stddev=0.5), dtype=tf.float32)
  b = tf.Variable(tf.random_normal((column_num, 1), mean=0.0, stddev=0.5), dtype=tf.float32)
  return w, b

def cross_op(x0, x, w, b):
  # x0 and x, shape mxd ; # w  and b, shape dx1
  #print ('x0 in cross_op', x0.get_shape())
  #print ('x  in corss_op', x.get_shape())
  #print ('w  in cross_op', w.get_shape())
  #print ('b  in cross_op', b.get_shape())
  x0 = tf.expand_dims(x0, axis=2) # mxdx1
  x  = tf.expand_dims(x,  axis=2) # mxdx1
  #print ('x0 in cross_op, after expand_dims', x0.get_shape())
  #print ('x  in cross_op, after expand_dims', x.get_shape())
  multiple = w.get_shape().as_list()[0]

  x0_broad_horizon = tf.tile(x0, [1,1,multiple])   # mxdx1 -> mxdxd #
  #print ('x0_broad in cross_op, after tf.tile', x0_broad_horizon.get_shape())
  x_broad_vertical = tf.transpose(tf.tile(x,  [1,1,multiple]), [0,2,1]) # mxdx1 -> mxdxd #
  #print ('x_broad_vertical, after tf.tile and tf.trans', x_broad_vertical.get_shape())
  w_broad_horizon  = tf.tile(w,  [1,multiple])     # dx1 -> dxd #
  #print ('w_broad_horizon, after tf.tile', w_broad_horizon.get_shape())
  mid_res = tf.multiply(tf.multiply(x0_broad_horizon, x_broad_vertical), w) # mxdxd # here use broadcast compute # 
  #print ('mid_res, after multiply x0_broad, x_broad, w', mid_res.get_shape())
  res = tf.reduce_sum(mid_res, axis=2) # mxd #
  #print ('res, after tf.reduce_sum(2)', res.get_shape())
  res = res + tf.transpose(b) # mxd + 1xd # here also use broadcast compute #a
  #print ('res, after +', res.get_shape())
  #print ('-----------------------------------')
  return res

def cross_op2(x0, x, w, b):
  ## for cicle compute ## ??????????????????????????????????????????????????????Batch_size????????? ##
  batch_num = x0.get_shape().as_list()[0]
  #print ('in cross_op2, tf.shape(x0)', tf.shape(x0))
  #print ('in cross_op2, x0.get_shape()', x0.get_shape())
  res = []
  for i in range(batch_num):
    dd = tf.matiply(x0[i,:,:], tf.transpose(x[i,:,:]))
    dc = tf.matmul(dd, w) + b
    res[i] = dc
  return res + x0

def Mode_mine(features, labels, mode, params):
  columns = build_model_columns()
  ## from input to dense x0 ## if need stack not-embedding and embedding ##
  input_layer = tf.feature_column.input_layer(features = features, feature_columns = columns)
  ## cross part ##
  print ('features.shape', features)
  column_num   = input_layer.get_shape().as_list()[1]
  print ('column_num, before cross_variable_creat: ', column_num)
  c_w_1, c_b_1 = cross_variable_creat(column_num)
  c_w_2, c_b_2 = cross_variable_creat(column_num)
  #c_w_3, c_b_3 = cross_variable_creat(column_num)
  #c_w_4, c_b_4 = cross_variable_creat(column_num)
  #c_w_5, c_b_5 = cross_variable_creat(column_num)
  c_layer_1 = cross_op(input_layer, input_layer, c_w_1, c_b_1) + input_layer
  c_layer_2 = cross_op(input_layer, c_layer_1,   c_w_2, c_b_2) + c_layer_1
  c_layer_5 = c_layer_2
  #c_layer_3 = cross_op(input_layer, c_layer_2,   c_w_3, c_b_3) + c_layer_2
  #c_layer_4 = cross_op(input_layer, c_layer_3,   c_w_4, c_b_4) + c_layer_3
  #c_layer_5 = cross_op(input_layer, c_layer_4,   c_w_5, c_b_5) + c_layer_4
  ## deep part ##
  h_layer_1 = tf.layers.dense(inputs = input_layer, units = 50, \
                              activation = tf.nn.relu, \
                              use_bias   = True)#, \
  bn_layer_1 = tf.layers.batch_normalization(inputs = h_layer_1, axis = -1, \
                              momentum   = 0.99, \
                              epsilon    = 0.001, \
                              center     = True, \
                              scale      = True)
  h_layer_2 = tf.layers.dense(inputs = bn_layer_1, units = 40, \
                              activation = tf.nn.relu, \
                              use_bias   = True)
  ## h_layer_2 = [mxd]; c_layer_5 = [mxd_hidden] ## ?????????????????? #
  m_layer   = tf.concat([h_layer_2, c_layer_5], 1)
  o_layer   = tf.layers.dense(inputs = m_layer, units=1, \
                              activation = None, \
                              use_bias   = True)
  o_prob    = tf.nn.sigmoid(o_layer)
  print ('in model_mine, o_prob.get_shape', o_prob.get_shape())
  print ('in model_mine, labels.get_shape', labels.get_shape())
  predictions     = tf.cast((o_prob>0.5), tf.float32)
  labels          = tf.cast(labels, tf.float32)
  prediction_dict = {'income_bracket':predictions}
  # ======= define your loss for the model ======= #
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=o_layer))
  accuracy  = tf.metrics.accuracy(labels, predictions)
  tf.summary.scalar('accuracy', accuracy[1])
  ## something is not clear about accuracy ##
  # ======= define your train_op for the model ======= #
  optimizer = tf.train.AdamOptimizer(learning_rate=0.0004, beta1=0.9, beta2=0.999)
  train_op  = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
  # ======= last return ======= #
  if mode == tf.estimator.ModeKeys.TRAIN:
     return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
  elif mode == tf.estimator.ModeKeys.EVAL:
     return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy':accuracy})
  elif mode == tf.estimator.ModeKeys.PREDICT:
     return tf.estimator.EstimatorSpec(mode, predictions=predictions)
  else: print('ERROR')

def build_estimator(model_dir, model_type):
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))
  if model_type == 'deep_cross':
    return tf.estimator.Estimator(model_fn = Mode_mine,
        model_dir=model_dir,
#        feature_columns=deep_columns, # how to set this feature_columns here ? must be solved! #
        config=run_config)
  else: print ('error')

def input_fn(data_file, num_epochs, shuffle, batch_size):
  assert tf.gfile.Exists(data_file), ('no file named : '+str(data_file))

  def process_list_column(list_column):
    sparse_strings = tf.string_split(list_column, delimiter="|")
    return sparse_strings.values
  def parse_csv(value):
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    features['workclass'] = process_list_column([features['workclass']])
    labels = tf.equal(features.pop('income_bracket'), '>50K')
    labels = tf.reshape(labels, [-1])
    return features, labels

  dataset = tf.contrib.data.TextLineDataset(data_file)
  if shuffle: dataset = dataset.shuffle(buffer_size=5000)
  dataset = dataset.map(parse_csv, num_threads=5)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator  = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()

  return features, labels

def main(unused_argv):
  # Clean up the model directory if present
  shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
  model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

  # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
  for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    model.train(input_fn=lambda: input_fn(
        FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

    results = model.evaluate(input_fn=lambda: input_fn(
        FLAGS.test_data, 1, False, FLAGS.batch_size))

    # Display evaluation metrics
    print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
    print('-' * 60)

    for key in sorted(results):
      print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
