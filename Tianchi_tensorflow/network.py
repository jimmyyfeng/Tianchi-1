#encoding:utf-8
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import tensorflow.contrib.layers as ly
DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, batch_size=2,keep_prob=0.2,trainable=True, is_training=True):
        # The input nodes for this network
        self.inputs = inputs
        self.batch_size=batch_size
        self.keep_prob=keep_prob
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        self.is_training=is_training
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')
        
    def common_normalization(self, input,name=None, activation_fn=None, scale=True):
        
        output = slim.batch_norm(
            input,
            activation_fn=activation_fn,
            is_training=self.is_training,
            updates_collections=None,
            scale=scale
            )
        return output    
        
    @layer
    def merge(self,input,axis=3,name="merged"):
        return tf.concat(input,axis,name=name)
    @layer     
    def upsample2d(self,
             input,
             output_shape,
             ksize,
             stride,
             name="sampled",
             relu=True,
             padding=DEFAULT_PADDING,
             biased=True,
             norm=True):
        self.validate_padding(padding)
        with tf.variable_scope(name) as scope:
            filter = tf.get_variable('weight', [ksize, ksize, output_shape[-1], input.get_shape()[-1]],dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0,0.02))
            result=tf.nn.conv2d_transpose(input, filter, output_shape=output_shape,
                                strides=[1, stride, stride, 1])
            biases = tf.get_variable('biases', [output_shape[-1]], dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            result=tf.nn.bias_add(result, biases)
            if norm:
                result=self.common_normalization(result)
            if relu:
                result=tf.nn.relu(result)
            return result
    @layer     
    def upsample3d_v2(self,
             input,
             co,
             ksize,
             stride,
             name="upsampled",
             relu=True,
             padding=DEFAULT_PADDING,
             biased=True,
             norm=True):
        self.validate_padding(padding)
        with tf.variable_scope(name) as scope:
            inlist=input.get_shape().as_list()
            output_shape=[self.batch_size,inlist[1]*2,inlist[2]*2,inlist[3]*2,co]
            print output_shape
            filter = tf.get_variable('weight', [ksize, ksize,ksize, co, input.get_shape()[-1]],dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0,0.02))
            result=tf.nn.conv3d_transpose(input, filter, output_shape=output_shape,
                                strides=[1, stride, stride,stride, 1])
            biases = tf.get_variable('biases', [co], dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            result=tf.nn.bias_add(result, biases)
            if norm:
                result=self.common_normalization(result)
            if relu:
                result=tf.nn.relu(result)
            return result
    
    @layer     
    def upsample3d(self,
             input,
             output_shape,
             ksize,
             stride,
             name="upsampled",
             relu=True,
             padding=DEFAULT_PADDING,
             biased=True,
             norm=True):
        self.validate_padding(padding)
        with tf.variable_scope(name) as scope:
            filter = tf.get_variable('weight', [ksize, ksize,ksize, output_shape[-1], input.get_shape()[-1]],dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0,0.02))
            result=tf.nn.conv3d_transpose(input, filter, output_shape=output_shape,
                                strides=[1, stride, stride,stride, 1])
            biases = tf.get_variable('biases', [output_shape[-1]], dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            result=tf.nn.bias_add(result, biases)
            if norm:
                result=self.common_normalization(result)
            if relu:
                result=tf.nn.relu(result)
            return result
    def pool_common(self,input,ksize,stride,dimensions=2,relu=False,padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        if dimensions==2:
            fn=tf.nn.max_pool
        else:
            fn=tf.nn.max_pool3d
        return fn(input,
                              ksize=[1]+[ksize]*dimensions+[1],
                              strides=[1]+[stride]*dimensions+[1],
                              padding=padding,
                              )
        
    def conv_common(self,input,co,ksize,stride,name,dimensions=2,relu=True,padding=DEFAULT_PADDING,biased=True,norm=True):
        self.validate_padding(padding)
        if dimensions==2:
            fn=tf.nn.conv2d
        else:
            fn=tf.nn.conv3d
        with tf.variable_scope(name) as scope:
            filter=tf.get_variable('weights', [ksize]*dimensions+[ input.get_shape()[-1], co],dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            strides=[1]+[stride]*dimensions+[1]
            result=fn(input, filter, strides=strides, padding=padding)
            biases = tf.get_variable('biases', [co],dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            result=tf.nn.bias_add(result,biases)
            if norm:
                result=self.common_normalization(result)
            if relu:
                result=tf.nn.relu(result)
            return result
    @layer
    def Inception_v2(self,input,co,name,dimensions=2,relu=True,padding=DEFAULT_PADDING,biased=True,norm=True):
        assert(dimensions in (2,3))
        assert(co%4==0)
        cos=[co/4]*4
        self.validate_padding(padding)
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('branch1'):                
                result_1 = self.conv_common(input,cos[0],1,1,name="conv1",dimensions=dimensions,relu=False,norm=False)
            with tf.variable_scope('branch2'):               
                result_2 = self.conv_common(input,2*cos[1],1,1,name="conv1",dimensions=dimensions,relu=True)
                result_2 = self.conv_common(result_2,cos[1],3,1,name="conv2",dimensions=dimensions,relu=False,norm=False)
            with tf.variable_scope('branch3'):
                result_3 = self.conv_common(input,2*cos[2],1,1,name="conv1",dimensions=dimensions,relu=True)
                result_3 = self.conv_common(result_3,cos[2],5,1,name="conv2",dimensions=dimensions,relu=False,norm=False)
            with tf.variable_scope('branch4'):
                result_4 = self.pool_common(input,2,1,dimensions=dimensions)
                result_4 = self.conv_common(result_4,cos[3],1,1,name="conv1",dimensions=dimensions,relu=False,norm=False)
            print len(input.get_shape())-1
            result=tf.concat([result_1,result_2,result_3,result_4],len(input.get_shape())-1,name=name)
            if norm:
                result=self.common_normalization(result)
            if relu:
                result=tf.nn.relu(result)
            return result
    @layer
    def spatial_red_block(self,input,name,dimensions=2,relu=True,padding=DEFAULT_PADDING,biased=True,norm=True):
    
    #空间压缩模块，经过此模块后input的大小减半，通道数加倍
    # 分支1：2*2，stride=2的最大池化
    # 分支2:3*3，stride=2的卷积
    # 分支3:1*1，stride=1的卷积，3*3，stride=2的卷积
    # 分支4：1*1，stride=1的卷积，3*3，stride=1的卷积，3*3，stride=2的卷积
    
        assert(dimensions in (2,3)),'dimensions must be in (2,3)'
        assert(name!=''),'name must be not null '
        in_shape=input.get_shape().as_list()
        co=2*in_shape[-1]
        assert(co%16==0),'the input channel must be divided by 16'
        self.validate_padding(padding)
        # out_list=[4*co/16,5*]
        with tf.variable_scope(name) as scope:
            with tf.variable_scope("branch1"):
                result_1=self.pool_common(input,2,2,dimensions=dimensions)
            with tf.variable_scope("branch2"):
                result_2 = self.conv_common(input,co/4,3,2,name="conv1",dimensions=dimensions,relu=False,norm=False)
            with tf.variable_scope("branch3"):
                result_3 = self.conv_common(input,co/4,1,1,name="conv1",dimensions=dimensions,relu=True)
                result_3 = self.conv_common(result_3,co*5/16,3,2,name="conv2",dimensions=dimensions,relu=False,norm=False)
            with tf.variable_scope("branch4"):
                result_4 = self.conv_common(input,co/4,1,1,name="conv1",dimensions=dimensions,relu=True)
                result_4 = self.conv_common(result_4,co*5/16,3,1,name="conv2",dimensions=dimensions,relu=True,norm=True)
                result_4 = self.conv_common(result_4,co*7/16,3,2,name="conv3",dimensions=dimensions,relu=False,norm=False)
            result=tf.concat([result_1,result_2,result_3,result_4],len(input.get_shape())-1,name=name)
            if norm:
                result=self.common_normalization(result)
            if relu:
                result=tf.nn.relu(result)
            return result
            
                
    @layer
    def feat_red(self,input,name,dimensions=2,relu=True,padding=DEFAULT_PADDING,biased=True,norm=True):
        #'''
        #特征压缩模块，经过此模块后空间大小不变，通道数减半
        #'''
        assert(dimensions in (2,3)),'dimensions must be in (2,3)'
        assert(name!=''),'name must be not null '
        in_shape=input.get_shape().as_list()
        assert(in_shape[-1]%2==0),'the input channel must be divided by 2'
        co=in_shape[-1]/2
        with tf.variable_scope(name) as scope:
            result=self.conv_common(input,co,1,1,name="conv1",dimensions=dimensions,relu=False,norm=False)
            if norm:
                result=self.common_normalization(result)
            if relu:
                result=tf.nn.relu(result)
            return result
    @layer
    def res_conc_block(self,input,cn,name,dimensions=2,relu=True,padding=DEFAULT_PADDING,biased=True,norm=True):
    #'''
    #残差链接模块
    #分支1：3*3，stride=1的卷积
    #分支2:1*1，stride=1的卷积，3*3，stride=1的卷积
    #分支3：1*1，stride=1的卷积，3*3，stride=1的卷积，3*3，stride=1的卷积
    #分支1,2,3concat到一起，1*1，stride=1卷积
    #最后在与input相加
    #'''
        assert(dimensions in (2,3)),'dimensions must be in (2,3)'
        assert(name!=''),'name must be not null '
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('branch1'): 
                result_1 = self.conv_common(input,cn,3,1,name="conv1",dimensions=dimensions,relu=False,norm=False)
            with tf.variable_scope("branch2"):
                result_2 = self.conv_common(input,cn,1,1,name="conv1",dimensions=dimensions)
                # result_2 = self.conv_common(result_2,cn,3,1,name="conv2",dimensions=dimensions)
                result_2 = self.conv_common(result_2,cn,3,1,name="conv2",dimensions=dimensions,relu=False,norm=False)
            with tf.variable_scope("branch3"):
                result_3 = self.conv_common(input,cn,1,1,name="conv1",dimensions=dimensions)
                result_3 = self.conv_common(result_3,cn,3,1,name="conv2",dimensions=dimensions)
                result_3 = self.conv_common(result_3,cn,3,1,name="conv3",dimensions=dimensions,relu=False,norm=False)
            tmp_result=tf.concat([result_1,result_2,result_3],len(input.get_shape())-1,name=name)
            tmp_result=tf.nn.relu(self.common_normalization(tmp_result))
            result = self.conv_common(tmp_result,input.get_shape()[-1],1,1,name="conv1",dimensions=dimensions,relu=True,norm=True)
            result=result+input
            # if norm:
            #     result=self.common_normalization(result)
            # if relu:
            #     result=tf.nn.relu(result)
            return result
                
            
        
        
    
    @layer
    def Inception_v1(self,input,co,name,dimensions=2,relu=True,padding=DEFAULT_PADDING,biased=True,norm=True):
        assert(dimensions in (2,3))
        assert(co%4==0)
        cos=[co/4]*4
        self.validate_padding(padding)
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('branch1'):                
                result_1 = self.conv_common(input,cos[0],1,2,name="conv1",dimensions=dimensions,relu=False,norm=False)
            with tf.variable_scope('branch2'):               
                result_2 = self.conv_common(input,2*cos[1],1,1,name="conv1",dimensions=dimensions,relu=True)
                result_2 = self.conv_common(result_2,cos[1],3,2,name="conv2",dimensions=dimensions,relu=False,norm=False)
            with tf.variable_scope('branch3'):
                result_3 = self.conv_common(input,2*cos[2],1,1,name="conv1",dimensions=dimensions,relu=True)
                result_3 = self.conv_common(result_3,cos[2],5,2,name="conv2",dimensions=dimensions,relu=False,norm=False)
            with tf.variable_scope('branch4'):
                result_4 = self.pool_common(input,2,2,dimensions=dimensions)
                result_4 = self.conv_common(result_4,cos[3],1,1,name="conv1",dimensions=dimensions,relu=False,norm=False)
            print len(input.get_shape())-1
            result=tf.concat([result_1,result_2,result_3,result_4],len(input.get_shape())-1,name=name)
            if norm:
                result=self.common_normalization(result)
            if relu:
                result=tf.nn.relu(result)
            return result
                
                
    @layer
    def ResLy_2(self,input,co,stride,name,dimensions=2,relu=True,padding=DEFAULT_PADDING,biased=True,norm=True):
        #ResNet 单元，包含两个3*3卷基层
        #第一个卷基层输出为co1,
        #两个卷基层步长相同，都为stride
        #步长stride只能为1
        assert(stride==1)
        assert(dimensions in (2,3))
        self.validate_padding(padding)
        if dimensions==2:
            fn=tf.nn.conv2d
        else:
            fn=tf.nn.conv3d
        with tf.variable_scope(name) as scope:
            
            result_1 = self.conv_common(input,co,3,1,name="conv1",dimensions=dimensions,relu=True,norm=True)
            result_2 = self.conv_common(result_1,input.get_shape()[-1],3,1,name="conv2",dimensions=dimensions,relu=False,norm=False)
            result=result_2+input
            if norm:
                result=self.common_normalization(result)
            if relu:
                result=tf.nn.relu(result)
            return result
    @layer
    def ResLy_3(self,input,co1,co2,stride,name,dimensions=2,relu=True,padding=DEFAULT_PADDING,biased=True):
        #ResNet 单元，包含两个1*1卷基层和一个3*3
        #第一个卷基层输出为co1,，第二个输出为co2
        #两个卷基层步长相同，都为stride
        #步长stride只能为1
        assert(stride==1)
        assert(dimensions in (2,3))
        self.validate_padding(padding)
        if dimmension==2:
            fn=tf.nn.conv2d
        else:
            fn=tf.nn.conv3d
        
        with tf.variable_scope(name) as scope:
            filter1=tf.get_variable('weights_1', [1]*dimensions + [input.get_shape()[-1], co1],dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            strides=[1]+[stride]*dimensions+[1]
            result=fn(input, filter1, strides=strides, padding=padding)
            biases = tf.get_variable('biases_1', [co1],dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            result = tf.nn.bias_add(result, biases)
            result=tf.nn.relu(result)
                
            filter2=tf.get_variable('weights_2', [3]*dimensions+[co1,co2],dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            result=fn(result, filter2, strides=strides, padding=padding)
            biases = tf.get_variable('biases_2', [ co2],dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            result = tf.nn.relu(tf.nn.bias_add(result, biases))
            
            
            filter3=tf.get_variable('weights_3', [1]*dimensions+[ co2, input.get_shape()[-2]],dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            result=fn(result, filter3, strides=strides, padding=padding)
            biases = tf.get_variable('biases_3', [ input.get_shape()[-2]],dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            result = tf.nn.bias_add(result, biases)
            result =tf.add(result,input)
            if relu:
                result=tf.nn.relu(result)
            
            
            return result
        
    @layer
    def conv(self,
             input,
             co,
             ksize,
             stride,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True,
             norm=True):
        self.validate_padding(padding)
        with tf.variable_scope(name) as scope:
            filter =  tf.get_variable('weights', [ksize, ksize, input.get_shape()[-1], co],dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            strides=[1,stride,stride,1]
            result=tf.nn.conv2d(input, filter, strides=strides, padding=padding)
            biases = tf.get_variable('biases', [co],dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            result = tf.nn.bias_add(result, biases)
            if norm:
                result=self.common_normalization(result)
            if relu:
                result=tf.nn.relu(result)
            return result
    
    @layer
    def conv3d(self,
             input,
             co,
             ksize,
             stride,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True,
             norm=True):
        self.validate_padding(padding)
        with tf.variable_scope(name) as scope:
            filter=tf.get_variable('weights', [ksize,ksize, ksize, input.get_shape()[-1], co],dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            strides=[1,stride,stride,stride,1]
            result=tf.nn.conv3d(input, filter, strides=strides, padding=padding)
            biases = tf.get_variable('biases', [co],dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            result = tf.nn.bias_add(result, biases)
            if norm:
                result=self.common_normalization(result)
            if relu:
                result=tf.nn.relu(result)
            return result

    @layer
    def atrous_conv(self,
                    input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output
        
    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)
    @layer
    def sigmoid(self,input,name):
        return tf.sigmoid(input,name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)
    @layer
    def max_pool3d(self, input, k_h, k_w, k_c,s_h, s_w,s_c ,name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool3d(input,
                              ksize=[1, k_h, k_w, k_c,1],
                              strides=[1, s_h, s_w, s_c,1],
                              padding=padding,
                              name=name)


    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def reshape(self,input,num,name="flatten"):
        return tf.reshape(input,[-1,num])
    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)
    @layer
    def flatten(self,input,name="flaggen"):
        return tf.reshape(input,[input.shape[0],-1])

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        return tf.nn.softmax(input, name=name)
        
    @layer
    def batch_normalization(self, input, name, is_training, activation_fn=None, scale=True):
        with tf.variable_scope(name) as scope:
            output = slim.batch_norm(
                input,
                activation_fn=activation_fn,
                is_training=False,
                updates_collections=None,
                scale=scale,
                scope=scope)
            return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)
