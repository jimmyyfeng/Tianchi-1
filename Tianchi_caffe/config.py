#encoding:utf-8
import time
tfmt = '%m%d_%H%M%D'
class DefaultConfig:
    '''
    不要在这里修改
    '''
    #############################################
    #数据路径
    ##############################################
    online=False
    if online==False:
        data_root = '/home/x/data/datasets/tianchi/' # 数据保存路径
        candidate_center="/home/x/dcsb/Tianchi_pytorch/csv/center.csv"#分割网络产生的疑似结点中心位置保存路径
        all_annotation = "/home/x/dcsb/Tianchi_pytorch/csv/annotations.csv"#训练集加验证集的标注信息
        mhd_information='/home/x/dcsb/refactor/del/information.csv'#此csv文件保存了全部样本的原点以及spacing信息
        data_train='/home/x/data/datasets/tianchi/train/'#全部原始训练样本
        nodule_cubic='/mnt/7/train_nodule_cubic/'#从训练样本上切下的结点立方体保存路径
        candidate_cubic='/mnt/7/0705_train_48_64_candidate/'#从训练样本上切下的候选结点立方体保存路径
        save_dir='/mnt/7/0821_caffe_64_80/'#测试分割网络块保存路径
        train_txt="/home/x/data/datasets/tianchi/txtfiles/train.txt"
        cls_data_root='/home/x/dcsb/Tianchi_caffe/'
    
    else:
        data_root='/workspace/pai/data/'
        candidate_center="/workspace/pai/data/csv_file/center.csv"#分割网络产生的疑似结点中心位置保存路径
        all_annotation = "/workspace/pai/data/csv_file/annotations.csv"
        mhd_information='/workspace/pai/data/csv_file/mhd_information.csv'#此csv文件保存了全部样本的原点以及spacing信息
        #data_train='/home/x/data/datasets/tianchi/train/'#全部原始训练样本
        nodule_cubic='/workspace/pai/data/nodule_cubic/'#从训练样本上切下的结点立方体保存路径
        candidate_cubic='/workspace/pai/data/candidate_cubic/'#从训练样本上切下的候选结点立方体保存路径
        save_dir='/workspace/pai/data/caffe_64_80/'#测试分割网络块保存路径
        train_txt="/workspace/pai/data/txtfiles/train.txt"
        
     
    
    annotatiion_csv = "/home/x/dcsb/Tianchi_pytorch/csv/annotations.csv"
    candidate_center="/home/x/dcsb/Tianchi_pytorch/csv/center.csv"#分割网络产生的疑似结点中心位置保存路径
    data_train='/home/x/data/datasets/tianchi/train/'#全部原始训练样本
    nodule_cubic='/mnt/7/train_nodule_cubic/'#从训练样本上切下的结点立方体保存路径
    seg_ratio=2
    train_crop_size=48
    candidate_cubic='/mnt/7/0705_train_48_64_candidate/'#从训练样本上切下的候选结点立方体保存路径
    #################################################
    #模型保存路径
    ###################################################
    if online==False:
        model_seg_weight="snashots/mnist_iter_3750.caffemodel"#用于测试分割网络的模型权重
        model_seg_pre_weight="snashots/seg_0831_iter_622.caffemodel"#训练分割网络时预加载 的模型权重
        
    
    
    ################################################
    #   Prototxt文件路径
    ################################################
    if online==False:
        model_def_seg_train="prototxt/segmentation_train.prototxt"#分割网络训练prototxt
        model_def_seg_test1="prototxt/segmentation_test1.prototxt"#分割网络测试prototxt  块大小 80
        model_def_seg_test2="prototxt/segmentation_test2.prototxt"#分割网络测试prototxt  块大小 64
        model_seg_solver="solver/seg_solver.prototxt"#分割网络solver文件路径
        
    else:
        model_def_seg_train="/workspace/pai/prototxt/segmentation_train.prototxt"#分割网络训练prototxt
        model_def_seg_test1="/workspace/pai/prototxt/segmentation_test1.prototxt"#分割网络测试prototxt  块大小 80
        model_def_seg_test2="/workspace/pai/prototxt/segmentation_test2.prototxt"#分割网络测试prototxt  块大小 64
        model_seg_solver="/workspace/pai/solver/seg_solver.prototxt"#分割网络solver文件路径
    ################################################
    #训练参数
    ################################################
    if online:
        batch_size=32
    else:
        batch_size_train_seg=4
        batch_size_test_seg=1
    
    
    
class Config(DefaultConfig):
    '''
    在这里修改,覆盖默认值
    '''
opt = Config()