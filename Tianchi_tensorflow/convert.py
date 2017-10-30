from glob import glob
from scipy import misc
import numpy as np
img_list=glob("/home/x/dc/remote_file/data/TianChi/img/*.png")

masks_list=[img_file.replace('img','mask') for img_file in img_list]
img_data=np.zeros((len(img_list),512,512))
mask_data=np.zeros(img_data.shape)

for i in range(len(img_list)):
    tmp_img=misc.imread(img_list[i])
    if tmp_img.shape[0]!=512:
        tmp_img=misc.imresize(tmp_img,(512,512))
    img_data[i]=tmp_img
    tmp_mask=misc.imread(masks_list[i])
    if tmp_mask.shape[0]!=512:
        tmp_mask=misc.imresize(tmp_mask,(512,512))        
    mask_data[i]=tmp_mask
    
np.save("/home/x/dc/remote_file/data/TianChi/imgs1.npy",img_data)
np.save("/home/x/dc/remote_file/data/TianChi/masks1.npy",mask_data)