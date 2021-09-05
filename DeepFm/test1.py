# linenum=0
# with open ('F:/py_project/DeepFmByHai3/Data/pred2.csv', 'r') as f:
#     with open ('F:/py_project/DeepFmByHai3/Data/lables.txt', 'w') as fw:
#         for line in f:
#             fw.write(line)

# with open ('F:/py_project/DeepFmByHai3/Data/w.txt', 'r') as fw:
#     w=fw.read()
#     wlist=w.split()
#     print(len(wlist))
#     print (wlist)
#     print (float (wlist[0]))
# with open ('F:/py_project/DeepFmByHai3/Data/b.txt', 'r') as fb:
#     b=fb.read()
#     blist=b.split()
#     print (len (blist))
#
# list1 = list(map(lambda a,b:float(a)*float(b),inputlist,wlist))
#
# sum=sum(list1)+float(blist[0])

def calAUC(prob,labels):
    f = list(zip(prob,labels))
    rank = [values2 for values1,values2 in sorted(f,key=lambda x:x[0])]
    rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if(labels[i]==1):
            posNum+=1
        else:
            negNum+=1
    auc = 0
    auc = (sum(rankList)- (posNum*(posNum+1))/2)/(posNum*negNum)
    print(auc)
    return auc

with open ('F:/py_project/DeepFmByHai3/Data/pred_cpu_fpga_te.csv', 'r') as f:
    input = f.read ()
    pre = input.split ()
    pre = list(map(lambda a:float(a),pre))
    print(pre)
    print(len(pre))

with open ('F:/py_project/DeepFmByHai3/Data/pred1.csv', 'r') as f:
    input = f.read ()
    true = input.split()
    true = list(map (lambda a: int(a), true))
    print (true)
    print (len (true))
calAUC(pre,true)



