import numpy
import pandas as pd
# linenum=0
# with open ('F:/py_project/DeepFmByHai3/Data/tr1_new.libsvm', 'r') as f:
#     with open ('F:/py_project/DeepFmByHai3/Data/tr11_new.libsvm', 'w') as f1:
#         with open ('F:/py_project/DeepFmByHai3/Data/tr12_new.libsvm', 'w') as f2:
#             with open ('F:/py_project/DeepFmByHai3/Data/tr13_new.libsvm', 'w') as f3:
#                 for line in f:
#                     linenum=linenum+1
#                     if linenum%3==0:
#                         f1.write(line)
#                     elif linenum%2==0:
#                         f2.write(line)
#                     else:
#                         f3.write (line)

# linenum=0
# with open ('F:/py_project/DeepFmByHai3/Data/va1.libsvm', 'r') as f:
#     with open ('F:/py_project/DeepFmByHai3/Data/pre_val.libsvm', 'w') as f1:
#         for line in f:
#             linenum=linenum+1
#             if linenum<=1000000:
#                 f1.write(line)


#2699 9480 tr1
#300 0520  va1
#100 0000  te1

# with open ('F:/py_project/DeepFmByHai3/Data/test.txt', 'r') as f:
#     for line in f:
#         features = line.rstrip ('\n').split (',')
#         print (len (features))
#         if (len (features) < 41):
#             continue
#         feat_vals = []
        # for i in range (0, len (continous_features)):
        #     val = dists.gen (i, features[continous_features[i]])
        #     feat_vals.append (str (continous_features[i]) + ':' + "{0:.6f}".format (val).rstrip ('0').rstrip ('.'))
        # for i in range (0, len (categorial_features)):
        #     val = dicts.gen (i, features[categorial_features[i]]) + categorial_feature_offset[i]
        #     feat_vals.append (str (val) + ':1')
        # out.write ("{1}\n".format (' '.join (feat_vals)))

with open ('F:/py_project/DeepFmByHai3/Data/pred_cpu_fpga.csv', 'r') as f:
    with open ('F:/py_project/DeepFmByHai3/Data/pred_cpu_fpga_te.csv', 'w') as f1:
        for line in f:
            features = line.rstrip('\n').split (',')
            f1.write("%f\n"%float(features[1]))


# with open ('F:/py_project/DeepFmByHai3/Data/testf1.txt', 'w') as f1:
#     for i in range(10):
#         t=numpy.array([[1,2,3]])
#         numpy.savetxt (f1, t)