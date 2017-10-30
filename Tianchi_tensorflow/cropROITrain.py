


def crop_test():
    '''
    crop roi for test dataset. for each test data, first use unet to predict the probablity mask.
    then choose the point which has the max score from the mask .crop an area whose center point is 
    that we choose before
    '''
    file_list=glob("/home/x/dc/remote_file/data/TianChi/test/"+"*.mhd")
    luna ="/home/x/dc/remote_file/data/TianChi/"
    nodules=[]
    labels=[]
    for idx,file_name in enumerate(tqdm(file_list)):
        
        
        
        
def draw_labeled_bboxes(img, labels):
    copied = np.copy(img)
    bboxes = []
    # Iterate through all detected nodules
    for nodule_number in range(1, labels[1]+1):
        # Find pixels with each nodule_number label value
        nonzero = (labels[0] == nodule_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        width = np.max(nonzerox) - np.min(nonzerox)
        height = np.max(nonzerox) - np.max(nonzeroy)

        if width > 5 and height > 5:
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bboxes.append(bbox)
            # Draw the box on the image
            #cv2.rectangle(img, bbox[0], bbox[1], (10, 10, 10), 2)
            #copied = cv2.addWeighted(copied, 1.0, img, 1.0, 0.)

    # Return the image
    return copied, bboxes   

def crop_nodule(img, bbox):
    padding = 5
    y_start = np.clip(bbox['Ymin'] - padding, 0, 512)
    y_end = np.clip(bbox['Ymax'] + padding, 0, 512)
    x_start = np.clip(bbox['Xmin'] - padding, 0, 512)
    x_end = np.clip(bbox['Xmax'] + padding, 0, 512)
    cropped = img[y_start:y_end, x_start:x_end]
    cropped = cv2.resize(cropped, (50, 50))
    return cropped



#####################
#
# Helper function to get rows in data frame associated 
# with each file
def get_filename(case):
    global file_list
    for f in file_list:        
        if case in f:
            return(f)

def crop_train():
    '''
    crop roi from the train dataset. for each train img,crop the nodule area in a rectangle
    then reverse it in 3 ways to augment it as the positive samples.
    random choose 10 point as the area center,crop the area as the negative samples
    '''
    df_node = pd.read_csv("/home/x/dc/remote_file/data/TianChi/csv/train/"+"annotations.csv")
    df_node["file"] = df_node["seriesuid"].apply(get_filename)
    df_node = df_node.dropna()
    file_list=glob("/home/x/dc/remote_file/data/TianChi/train/"+"*.mhd")
    luna ="/home/x/dc/remote_file/data/TianChi/"
    nodules=[]
    labels=[]
    for fcount, img_file in enumerate(tqdm(file_list)):
        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
        if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 
            # load the data once
            itk_img = sitk.ReadImage(img_file) 
            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
            num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
            origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
            # go through all nodes (why just the biggest?)
            patient_nodules=[]
            for node_idx, cur_row in mini_df.iterrows():       
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]
                # just keep 3 slices
                center = np.array([node_x, node_y, node_z])   # nodule center
                v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
                slice=img_array[int(v_center[2]),:,:]
                x,y=int(v_center[0]),int(v_center[1])
                span=diam/spacing
                Xmin,Xmax,Ymin,Ymax=int(x-span[0]/2-1),int(x+span[0]/2+1),int(y-span[1]/2-1),int(y+span[1]/2+1),
                bbox={}
                bbox['Xmin']=Xmin
                bbox['Xmax']=Xmax
                bbox['Ymin']=Ymin
                bbox['Ymax']=Ymax
                true_croped=crop_nodule(slice,bbox)
                a1,a2,a3=reverse(true_croped)
                nodules.append(true_croped)
                nodules.append(a1)
                nodules.append(a2)
                nodules.append(a3)
                labels.append(1)
                labels.append(1)
                labels.append(1)
                labels.append(1)
            for i in range(1,10):
                x,y=np.random.randint(200,400),np.random.randint(200,400)
                Xmin,Xmax,Ymin,Ymax=int(x-10),int(x+10),int(y-10),int(y+10),
                bbox={}
                bbox['Xmin']=Xmin
                bbox['Xmax']=Xmax
                bbox['Ymin']=Ymin
                bbox['Ymax']=Ymax
                true_croped=crop_nodule(slice,bbox)
                nodules.append(true_croped)
                labels.append(0)
    print np.array(nodules).shape                
    np.save(luna+'nodule_train/cropped_train.npy', np.array(nodules))
    np.save(luna+'nodule_train/label_train.npy',np.array(labels))