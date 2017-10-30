from caffe.proto import caffe_pb2
s = caffe_pb2.SolverParameter()

path='/home/x/dcsb/Tianchi_caffe/'
solver_file=path+'solver/seg_solver.prototxt'

s.train_net = path+'prototxt/cls_multi_kernel_train.prototxt'
s.test_net.append(path+'prototxt/cls_multi_kernel_val.prototxt')
s.test_interval = 782  
s.test_iter.append(313) 
s.max_iter = 100000 

s.base_lr = 0.001 
s.momentum = 0.9
s.weight_decay = 1e-5
s.lr_policy = 'step'
s.stepsize=26067
s.gamma = 0.1
s.display = 782
s.snapshot = 7820
s.snapshot_prefix = 'shapshot'
s.type = "SGD"
s.solver_mode = caffe_pb2.SolverParameter.GPU

with open(solver_file, 'w') as f:
    f.write(str(s))