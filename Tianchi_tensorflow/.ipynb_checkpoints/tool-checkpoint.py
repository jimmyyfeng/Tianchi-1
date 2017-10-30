#encoding:utf-8
import SimpleITK as sitk
import torch
import pandas as pd
import numpy as np
import skimage.measure
import skimage.segmentation
import skimage.morphology
import skimage.filters
import scipy.ndimage
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import scipy
import scipy.ndimage
from scipy.ndimage.interpolation import rotate
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('tqdm 是一个轻量级的进度条小包。。。')
    tqdm = lambda x : x

luna="/home/x/dcsb/data/TianChi/"

def segment_HU_scan_frederic(x, threshold=-350):
    mask = np.copy(x)
    binary_part = mask > threshold
    selem1 = skimage.morphology.disk(8)
    selem2 = skimage.morphology.disk(2)
    selem3 = skimage.morphology.disk(13)

    for iz in xrange(mask.shape[0]):
        # fill the body part
        filled = scipy.ndimage.binary_fill_holes(binary_part[iz])  # fill body
        filled_borders_mask = skimage.morphology.binary_erosion(filled, selem1)
        mask[iz] *= filled_borders_mask


        mask[iz] = skimage.morphology.closing(mask[iz], selem2)
        mask[iz] = skimage.morphology.erosion(mask[iz], selem3)
        mask[iz] = mask[iz] < threshold

    return mask.astype(np.int8)
def voxel_2_world(arr,data_dir,file_name):
    df_node=pd.read_csv(data_dir+"csv/information.csv")
    df_min=df_node[df_node['seriesuid']==file_name]
    index= df_min.index[0]
    if type(arr)==list:
        arr=np.array(arr)
    # if type(arr==np.ndarray):
    #     assert(arr.shape[1]==3)
    origin=[df_min.at[index,'originX'],df_min.at[index,'originY'],df_min.at[index,'originZ']]
    spacing=[df_min.at[index,'spacingX'],df_min.at[index,'spacingY'],df_min.at[index,'spacingZ']]
    return arr*spacing+origin

def select(file,use):
    df=pd.read_csv("/home/x/dcsb/data/TianChi/csv/"+use+"/annotations.csv")
    return df[df['seriesuid']==file]

def get_ct(file_name):
    mhd=sitk.ReadImage(file_name)
    img_arr=sitk.GetArrayFromImage(mhd)
    origin = np.array(mhd.GetOrigin())
    spacing = np.array(mhd.GetSpacing())
    return img_arr,origin[::-1],spacing[::-1]
def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image
    
def get_filename(file_list, case):
        for f in file_list:
            if case in f:
                return (f)

def make_mask_for_patients_V1(file_name,a=0,width=64,rand=25,data_dir=luna):
    image,origin,spacing=get_ct(file_name)
#     image,new_spacing=resample(image,spacing)
    # image=normalize(image)
    df_node= pd.read_csv(data_dir + "csv/train/annotations.csv")
    name=file_name.split('/')[-1][:-4]
    df_node = df_node[df_node['seriesuid']==name]
    image_mask=np.zeros(image.shape,dtype=np.float32)
    nodule_centers=[]
    for index, nodule in df_node.iterrows():
        try:
            coord_z,coord_y,coord_x=nodule.coordZ, nodule.coordY, nodule.coordX
            nodule_center = np.array([coord_z,coord_y,coord_x])
            v_center = np.rint((nodule_center - origin) / spacing)

            v_center = np.array(v_center, dtype=int)
            nodule_centers.append(v_center)
            radius=nodule.diameter_mm/2
            span=np.round(radius/spacing)
            image_mask[np.clip(int(v_center[0]-span[0]),0,image.shape[0]):np.clip(int(v_center[0]+span[0]),0,image.shape[0]),\
                   np.clip(int(v_center[1]-span[1]),0,image.shape[1]):np.clip(int(v_center[1]+span[1]),0,image.shape[1]),\
                   np.clip(int(v_center[2]-span[2]),0,image.shape[2]):np.clip(int(v_center[2]+span[2]),0,image.shape[2])]=int(1)
        except:
            print f_name
    width=width
    assert(width%2==0)
    half=width/2
    # prob=np.random.random_sample()  
    nodule_num=len(nodule_centers)
    nodule_list=[]
    mask_list=[]
    for index in range(nodule_num):
        cubic_img=np.zeros([width,width,width],dtype=np.float32)
        cubic_mask=np.zeros([width,width,width],dtype=np.float32)
        # index=np.random.randint(0,nodule_num,1)[0]
        offset=np.random.randint(-1*rand,rand,3)
        v_center =nodule_centers[index]+offset
        v_center_min=v_center-half-3
        v_center_max=image.shape-v_center-half-3
        for i in range(3):#判断是否超出图像范围
            if v_center_min[i]<0:
                v_center[i]=v_center[i]-v_center_min[i]
            if v_center_max[i]<0:
                v_center[i]=v_center[i]+v_center_max[i]
        radius=nodule.diameter_mm/2
        span=np.round(radius/spacing)
        cubic_img[:,:,:]=image[int(v_center[0]-half):int(v_center[0]+half),\
                               int(v_center[1]-half):int(v_center[1]+half),\
                               int(v_center[2]-half):int(v_center[2]+half)]
        cubic_mask[:,:,:]=image_mask[int(v_center[0]-half):int(v_center[0]+half),\
                               int(v_center[1]-half):int(v_center[1]+half),\
                               int(v_center[2]-half):int(v_center[2]+half)]
        nodule_list.append(normalize(cubic_img))
        mask_list.append(cubic_mask)
    # else:#很大的几率返回负样本
        cubic_img=np.zeros([width,width,width],dtype=np.float32)
        cubic_mask=np.zeros([width,width,width],dtype=np.float32)
        z,x,y=image.shape
        center_z=np.random.randint(half+1,z-half-1,1)
        center_xy=np.random.randint(half+1,x-half-1,2)
        v_center=np.concatenate([center_z,center_xy])
        cubic_img[:,:,:]=image[int(v_center[0]-half):int(v_center[0]+half),\
                               int(v_center[1]-half):int(v_center[1]+half),\
                               int(v_center[2]-half):int(v_center[2]+half)]
        cubic_mask[:,:,:]=image_mask[int(v_center[0]-half):int(v_center[0]+half),\
                               int(v_center[1]-half):int(v_center[1]+half),\
                               int(v_center[2]-half):int(v_center[2]+half)]
        nodule_list.append(normalize(cubic_img))
        mask_list.append(cubic_mask)
    return nodule_list,mask_list


def cropBlocks_v0(img_new):
    num=np.array(img_new.shape)/32
    off=np.array(img_new.shape)-32*num
    off_min=off/2
    blocks=[]
    indexs=[]
    for i in range(num[0]-1):
        for j in range(num[1]-1):
            for k in range(num[2]-1):
                block=np.zeros([64,64,64],dtype=np.float32)
                block[:,:,:]=img_new[off_min[0]+i*32:off_min[0]+i*32+64,off_min[1]+j*32:off_min[1]+j*32+64,off_min[2]+k*32:off_min[2]+k*32+64]
                blocks.append(block)
                indexs.append(np.array([i,j,k]))
    return np.array(blocks),np.array(indexs)  
def cropBlocks(img_new):
    num=np.array(img_new.shape)/64
    off=np.array(img_new.shape)-64*num
    off_min=off/2
    blocks=[]
    indexs=[]
    for i in range(num[0]):
        for j in range(num[1]):
            for k in range(num[2]):
                block=np.zeros([64,64,64],dtype=np.float32)
                block[:,:,:]=img_new[off_min[0]+i*64:off_min[0]+i*64+64,off_min[1]+j*64:off_min[1]+j*64+64,off_min[2]+k*64:off_min[2]+k*64+64]
                blocks.append(block)
                indexs.append(np.array([i,j,k]))
    return np.array(blocks),np.array(indexs)
        
def world_2_voxel(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return np.round(voxel_coord)
def get_seg_all(img_arr):
    seg=np.zeros(img_arr.shape)
    for i in range(img_arr.shape[0]):
        seg[i]=get_segmentation(img_arr[i])
    return seg
def resample(image, old_spacing, new_spacing=[1, 1, 1]):
        resize_factor = old_spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = old_spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        return image, new_spacing




def voxel_2_world_by_files(file_name,coord,use):
    '''
    file_name:文件名
    coord:像素坐标
    use:测试？验证？训练？ ‘test/’,'validation/','train/'
    '''
    luna="/home/x/dc/remote_file/data/TianChi/"
    mhd=sitk.ReadImage(luna+use+file_name+'.mhd')
    origin = np.array(mhd.GetOrigin())
    spacing = np.array(mhd.GetSpacing())
    return coord*spacing+origin