#encoding:utf-8
from glob import glob
import random
nodule_cubic='/mnt/7/train_nodule_cubic/'#从训练样本上切下的结点立方体保存路径
candidate_cubic='/mnt/7/0705_train_48_64_candidate/'#从训练样本上切下的候选结点立方体保存路径
nodule_list=glob(nodule_cubic+"*.npy")
random.shuffle(nodule_list)
candidate_list=glob(candidate_cubic+"*.npy")
random.shuffle(candidate_list)
nodule_len=len(nodule_list)
candidate_len=len(candidate_list)
with open("train_cls.txt","w") as f:
    for ii in range(int(nodule_len*0.8)):
        f.write(nodule_list[ii]+" 1"+ "\n")
    for jj in range(int(candidate_len*0.8)):
        file=candidate_list[jj]
        isnodule=file.split('/')[-1].split('_')[-1][0]
        if isnodule=='1':
            f.write(file+" 1"+"\n")
        else:f.write(file+" 0"+"\n")
f.close()

with open("val_cls.txt","w") as f1:
    for ii in range(int(nodule_len*0.8),nodule_len):
        f1.write(nodule_list[ii]+" 1"+ "\n")
    for jj in range(int(candidate_len*0.8),candidate_len):
        file=candidate_list[jj]
        isnodule=file.split('/')[-1].split('_')[-1][0]
        if isnodule=='1':
            f1.write(file+" 1"+"\n")
        else:f1.write(file+" 0"+"\n")
f1.close()
