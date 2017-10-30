from glob import glob
data_root='/home/x/data/datasets/tianchi/'
data_train=data_root+'train/'
train_list=glob(data_train+"*.mhd")
train_list.sort()
data_test=data_root+'test/'
test_list=glob(data_test+"*.mhd")
test_list.sort()
#data_val=data_root+'val/'
#val_list=glob(data_val+"*.mhd")
#val_list.sort()


def write_text(file_name,lists):
    file_object = open(file_name, 'w')
    for file in lists:
        file_object.write(file+ "\n") 
    file_object.close( )
write_text("train.txt",train_list)
write_text("test.txt",test_list)