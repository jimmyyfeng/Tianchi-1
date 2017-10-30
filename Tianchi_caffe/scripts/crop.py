import pandas as pd
from glob import  glob
from tqdm import tqdm
import numpy as np
from data.util import load_ct,check_center,load_ct_info
from config import opt
from multiprocessing import Pool
def is_exist(lists,f):
    for f_name in lists:
        if f in f_name:
            return True
    return False
df_node = pd.read_csv(opt.annotatiion_csv) 
train_list=glob('/home/x/data/datasets/tianchi/train/*.mhd')
print train_list[:10]
def crop_nodule(img_file):
    '''
    crop roi from the train dataset. for each train img,crop the nodule area in a rectangle
    then reverse it in 3 ways to augment it as the positive samples.
    random choose 10 point as the area center,crop the area as the negative samples
    '''

    file_name=img_file.split('/')[-1][:-4]
    print file_name
    mini_df = df_node[df_node["seriesuid"]==file_name]
    print mini_df
    if mini_df.shape[0]>0: 
        size,origin,spacing=load_ct_info(img_file) 
        if size[0]<108:
            crop_size=[size[0],108,108]
        else:crop_size=[108]*3
        patient_nodules=[]
        #import ipdb;ipdb.set_trace()
        for node_idx, cur_row in mini_df.iterrows():   
            #crop_img=np.zeros(crop_size)
            crop_mask=np.zeros(crop_size)
            diam = cur_row["diameter_mm"]
            center = np.array([cur_row["coordZ"], cur_row["coordY"], cur_row["coordX"]])   # nodule center
            v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
            v_center = check_center(crop_size,v_center,size)
            v_center=np.array(crop_size)/2
            #half=np.array(crop_size)/2
            #crop_img=img_arr[v_center[0]-half[0]:v_center[0]+crop_size[0]-half[0],\
            #                 v_center[1]-half[1]:v_center[1]+half[1],v_center[2]-half[2]:v_center[2]+half[2]]
            radius=diam/2
            span=np.round(radius/spacing).astype(np.int32)
            #print v_center
            #print span
            crop_mask[v_center[0]-span[0]:v_center[0]+span[0],v_center[1]-span[1]:v_center[1]+span[1],v_center[2]-span[2]:v_center[2]+span[2]]=1.0
            
            
            print crop_mask.sum()
            np.save('/mnt/7/train_nodule_mask/'+file_name+"_"+str(node_idx)+".npy", crop_mask)

if __name__=="__main__":
    #crop_nodule(train_list[0])
    pool=Pool(8)
    pool.map(crop_nodule,train_list)
