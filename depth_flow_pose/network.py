import numpy as np
import tensorflow as tf
#from vo_utils_dgx import compute_loss_photometric
slim = tf.contrib.slim
from .vo_utils import bilinear_sampler,getmeshgrid,bilinear_sampler_rweights

# ----------------------------------------------------------------------------------
# Commonly used layers and operations based on ethereon's implementation 
# https://github.com/ethereon/caffe-tensorflow
# Slight modifications may apply. FCRN-specific operations have also been appended. 
# ----------------------------------------------------------------------------------
# Thanks to *Helisa Dhamo* for the model conversion and integration into TensorFlow.
# ----------------------------------------------------------------------------------

DEFAULT_PADDING = 'SAME'
WEIGHT_DECAY = 0.0004

def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def interleave(tensors, axis):
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    return tf.reshape(tf.stack(tensors, axis + 1), new_shape)

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

    def __init__(self, inputs,is_training,batch):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        
        self.frontflows = []
        
        self.depth_preds = []
        
        self.summaries = []
        
        self.mean = tf.constant([91.090/255., 95.435/255.,  96.119/255.], dtype=tf.float32)
        # If true, the resulting variables are set as trainable
        self.trainable = is_training
        # Batch size needs to be set for the implementation of the interleaving
        self.batch_size = batch
        #print self.trainable
        self.setup(is_training)


    def setup(self,is_training):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=True):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''

        data_dict = np.load(data_path, encoding='latin1').item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iter(data_dict[op_name].items()):
                    
                    try:
                        if(param_name=='variance'):
                            param_name='moving_variance'
                        elif(param_name=='mean'):
                           param_name='moving_mean'
                        elif(param_name=='scale'):
                            param_name='gamma'
                        elif(param_name=='offset'):
                           param_name='beta'
                        if('fc' in op_name):
                               continue;
                            
                        var = tf.get_variable(param_name)
                        #print  var.name
                        session.run(var.assign(data))
                       
                    except ValueError:
                        #print op_name
                        #print "error"
                        if not ignore_missing:
                            raise
    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_layer_output(self, name):
        return self.layers[name]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, dtype = 'float32', initializer=tf.contrib.layers.xavier_initializer(), trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')
    
        
    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             biased=True,reuse=False):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid

        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name,reuse=reuse) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
            output = convolve(input, kernel)
           
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            else:
                if 'flow' in name:
                     
                    output = tf.nn.leaky_relu(output, name=scope.name)
            return output
        
        
    @layer
    def transpose_conv(self,
                       input,
                       k_h,
                       k_w,
                       c_o,
                       s_h,
                       s_w,
                       name,
                       padding=DEFAULT_PADDING):  
        batch = input.get_shape()[0]
        h =  input.get_shape()[1]
        w =  input.get_shape()[2]
        c_i = input.get_shape()[-1]
        
        deconv_shape = tf.stack([batch, h*2, w*2, c_o])
        deconv = lambda i, k: tf.nn.conv2d_transpose(i, k,deconv_shape,strides=[1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
                kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
                
                output = deconv(input,kernel)
                
                return output
            
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
            else:
                if 'flow' in name:
                    
                    output = tf.nn.leaky_relu(output, name=scope.name) 
            return output

    @layer
    def deconv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=False,
             leaky_relu = True,
             padding=DEFAULT_PADDING,
             biased=True, 
             reuse=False):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid

        # Convolution for a given input and kernel
        stride_shape = [1, s_h, s_w, 1]
        kernel_shape = [k_h, k_w, c_o, c_i]
        resize_shape = [1,2,2,0.5]
        input_shape = input.get_shape().as_list()
        out_shape = []
        
        
        if padding == 'SAME':
            out_shape.append(input_shape[0])
            out_shape.append(input_shape[1]*2)
            out_shape.append(input_shape[2]*2)
            out_shape.append(c_o)
        

            
        #print(out_shape)
            #out_shape.append(Math.floor((input_i + 2 * pad_i - kernel_i) / float(stride_i) + 1))
        deconv = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape=out_shape, strides=stride_shape, padding=padding)
        with tf.variable_scope(name, reuse=reuse) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_o, c_i])
            output = deconv(input, kernel)

            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if leaky_relu:
                # ReLU non-linearity
                output = tf.nn.leaky_relu(output,0.1, name=scope.name)
            return output   
    @layer
    def relu(self, input_data, name):
        return tf.nn.relu(input_data, name=name)

    @layer
    def max_pool(self, input_data, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input_data,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input_data, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input_data,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input_data, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input_data,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def bn_relu_conv(self,n_filters,filter_size=3,dropout_p=0.2,prefix=''):
        
        
        self.feed(self.batch_normalization(name = prefix+'_bn', is_training=self.trainable, activation_fn=tf.nn.relu))
        self.feed(self.conv(3, 3,n_filters, 1, 1, biased=True, relu=False, name=prefix+'_conv'))
        out = self.dropout(keep_prob=(1-dropout_p),name = prefix+'_drop')
        
        return out
    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input_data, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input_data.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input_data, [-1, dim])
            else:
                feed_in, dim = (input_data, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input_data, name):
        input_shape = map(lambda v: v.value, input_data.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input_data = tf.squeeze(input_data, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input_data, name)

    @layer
    def batch_normalization(self, input, name, is_training, activation_fn=None, scale=True,reuse=False):

        with tf.variable_scope(name,reuse=reuse) as scope:
           
            output = slim.batch_norm(
                input,
                decay = 0.999,
                epsilon=1e-5,
                reuse=None,
                activation_fn=activation_fn,
                is_training=is_training,
                updates_collections=None,
                scale=scale,
                scope=scope
                )
            return output

    @layer
    def batch_normalization_flow(self, input, name, is_training, activation_fn=None, scale=True,reuse=False):

        with tf.variable_scope(name,reuse=reuse) as scope:
           
            output = slim.batch_norm(
                input,
                decay = 0.9,
                epsilon=1e-5,
                reuse=None,
                activation_fn=activation_fn,
                is_training=is_training,
                updates_collections=None,
                scale=scale,
                scope=scope
                )
            return tf.nn.leaky_relu(output,alpha=0.1)

    @layer
    def dropout(self, input_data, keep_prob, name):
        return tf.nn.dropout(input_data, keep_prob, name=name)
    
    @layer
    def smart_cat(self,inputs,name,use_flow=False,compute_loss=False):

        activations1 = inputs[0]
        activations2 = inputs[1]
       
        if compute_loss:
            if use_flow :
                flow = inputs[2]
                shape = activations1.get_shape().as_list()
                
                img_coords = getmeshgrid(self.batch_size,shape[1],shape[2])
                img_coords = tf.transpose(img_coords, perm=[0, 2, 3, 1])
                #print np.shape(img_coords)
                img_coords = img_coords + flow
                #print np.shape(activations2)
                #print np.shape(img_coords)
                activations2.set_shape((self.batch_size,shape[1],shape[2],shape[3]))
                with tf.device('/cpu:0'):
                    self.summaries.append(tf.summary.histogram(name+'activations', activations1))
                activations1_,weights = bilinear_sampler_rweights(activations2,img_coords)
                with tf.device('/cpu:0'):
                    self.summaries.append(tf.summary.histogram(name+'activations_after', activations1_))
                #print np.shape(activations1_)
                
                diff = (activations1_ - tf.cast(tf.greater(weights,0),tf.float32) * activations1)
                cost = 0.01*tf.reduce_mean(tf.square(diff))
                self.coarse_costs.append(cost)
                return tf.concat(axis=3, values=([activations1,activations1_]), name=name)
            else :
                flow = inputs[2]
                shape = activations1.get_shape().as_list()
                
                img_coords = getmeshgrid(self.batch_size,shape[1],shape[2])
                img_coords = tf.transpose(img_coords, perm=[0, 2, 3, 1])
                #print np.shape(img_coords)
                img_coords = img_coords + flow
                #print np.shape(activations2)
                #print np.shape(img_coords)
                activations2.set_shape((self.batch_size,shape[1],shape[2],shape[3]))
               
                with tf.device('/cpu:0'):
                    self.summaries.append(tf.summary.histogram(name+'activations', activations1))
                activations1_,weights = bilinear_sampler_rweights(activations2,img_coords)
                with tf.device('/cpu:0'):
                    self.summaries.append(tf.summary.histogram(name+'activations_after', activations1_))
                #print np.shape(activations1_)
                
                diff = (activations1_ - tf.cast(tf.greater(weights,0),tf.float32) * activations1)
                cost = 0.01*tf.reduce_mean(tf.square(diff))
                self.coarse_costs.append(cost)
                return tf.concat(axis=3, values=([activations1,activations2]), name=name)
        else :
            if use_flow :
                flow = inputs[2]
                shape = activations1.get_shape().as_list()
                
                img_coords = getmeshgrid(self.batch_size,shape[1],shape[2])
                img_coords = tf.transpose(img_coords, perm=[0, 2, 3, 1])
                img_coords = img_coords + flow
                activations1_ = bilinear_sampler(activations2,img_coords)
                return tf.concat(axis=3, values=([activations1,activations1_]), name=name)
            else :
                return tf.concat(axis=3, values=([activations1,activations2]), name=name)
    
    
    def prepare_indices(self, before, row, col, after, dims ):


        x0, x1, x2, x3 = tf.meshgrid(before, row, col, after)
            
        x_0 = tf.reshape(x0,[-1]);
        x_1 = tf.reshape(x1,[-1]);
        x_2 = tf.reshape(x2,[-1]);
        x_3 = tf.reshape(x3,[-1]);
        

        linear_indices = x_3 + dims[3].value * x_2  + 2 * dims[2].value * dims[3].value * x_0 * 2 * dims[1].value + 2 * dims[2].value * dims[3].value * x_1
        linear_indices_int = tf.to_int32(linear_indices)

        return linear_indices_int

    def unpool_as_conv(self, size, input_data, id, stride = 1, ReLU = False, BN = True,reuse=False):

        # Convolution A (3x3)
        # --------------------------------------------------
        layerName = "layer%s_ConvA" % (id)
        self.feed(input_data)
        self.conv( 3, 3, size[3], stride, stride, name = layerName, padding = 'SAME', relu = False,reuse=reuse)
        outputA = self.get_output()

        # Convolution B (2x3)
        # --------------------------------------------------
        layerName = "layer%s_ConvB" % (id)
        padded_input_B = tf.pad(input_data, [[0, 0], [1, 0], [1, 1], [0, 0]], "CONSTANT")
        self.feed(padded_input_B)
        self.conv(2, 3, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False,reuse=reuse)
        outputB = self.get_output()

        # Convolution C (3x2)
        # --------------------------------------------------
        layerName = "layer%s_ConvC" % (id)
        padded_input_C = tf.pad(input_data, [[0, 0], [1, 1], [1, 0], [0, 0]], "CONSTANT")
        self.feed(padded_input_C)
        self.conv(3, 2, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False,reuse=reuse)
        outputC = self.get_output()

        # Convolution D (2x2)
        # --------------------------------------------------
        layerName = "layer%s_ConvD" % (id)
        padded_input_D = tf.pad(input_data, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
        self.feed(padded_input_D)
        self.conv(2, 2, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False,reuse=reuse)
        outputD = self.get_output()

        # Interleaving elements of the four feature maps
        # --------------------------------------------------
        left = interleave([outputA, outputB], axis=1)  # columns
        right = interleave([outputC, outputD], axis=1)  # columns
        Y= interleave([left, right], axis=2) # rows
        
        if BN:
            layerName = "layer%s_BN" % (id)
            self.feed(Y)
            self.batch_normalization(name = layerName, is_training=False, activation_fn=None,reuse=reuse)
            Y = self.get_output()

        if ReLU:
            Y = tf.nn.relu(Y, name = layerName)
        
        return Y


    def up_project(self, size, id, stride = 1, BN = True,reuse=False):
        
        # Create residual upsampling layer (UpProjection)

        input_data = self.get_output()

        # Branch 1
        id_br1 = "%s_br1" % (id)

        # Interleaving Convs of 1st branch
        out = self.unpool_as_conv(size, input_data, id_br1, stride, ReLU=True, BN=True,reuse=reuse)

        # Convolution following the upProjection on the 1st branch
        layerName = "layer%s_Conv" % (id)
        self.feed(out)
        self.conv(size[0], size[1], size[3], stride, stride, name = layerName, relu = False,reuse=reuse)

        if BN:
            layerName = "layer%s_BN" % (id)
            self.batch_normalization(name = layerName, is_training=False, activation_fn=None,reuse=reuse)

        # Output of 1st branch
        branch1_output = self.get_output()

            
        # Branch 2
        id_br2 = "%s_br2" % (id)
        # Interleaving convolutions and output of 2nd branch
        branch2_output = self.unpool_as_conv(size, input_data, id_br2, stride, ReLU=False,reuse=reuse)

        
        # sum branches
        layerName = "layer%s_Sum" % (id)
        output = tf.add_n([branch1_output, branch2_output], name = layerName)
        # ReLU
        layerName = "layer%s_ReLU" % (id)
        output = tf.nn.relu(output, name=layerName)

        self.feed(output)
        return self
    def up_project_flow(self, size, id, stride = 1, BN = True,reuse=False):
        
        # Create residual upsampling layer (UpProjection)

        input_data = self.get_output()

        # Branch 1
        id_br1 = "%s_br1" % (id)

        # Interleaving Convs of 1st branch
        out = self.unpool_as_conv(size, input_data, id_br1, stride, ReLU=False, BN=True,reuse=reuse)
        out = tf.maximum(out,0.1*out)

        # Convolution following the upProjection on the 1st branch
        layerName = "layer%s_Conv" % (id)
        self.feed(out)
        self.conv(size[0], size[1], size[3], stride, stride, name = layerName, relu = False,reuse=reuse)

        if BN:
            layerName = "layer%s_BN" % (id)
            self.batch_normalization(name = layerName, is_training=False, activation_fn=None,reuse=reuse)

        # Output of 1st branch
        branch1_output = self.get_output()

            
        # Branch 2
        id_br2 = "%s_br2" % (id)
        # Interleaving convolutions and output of 2nd branch
        branch2_output = self.unpool_as_conv(size, input_data, id_br2, stride, ReLU=False,reuse=reuse)

        
        # sum branches
        layerName = "layer%s_Sum" % (id)
        output = tf.add_n([branch1_output, branch2_output], name = layerName)
        # ReLU
        layerName = "layer%s_ReLU" % (id)
        output = tf.nn.leaky_relu(output,0.1)

        self.feed(output)
        return self
    
   
    
    def get_uncertainty(self,shapes_t):
                    
        

        alpha_beta_gamma =(self.get_layer_output('sigma_final'))
        

        a_list = []
        c_list=[]
        b_list = []

        for ii in range(0,self.batch_size):

          

                alpha_beta_gamma_i = tf.expand_dims(alpha_beta_gamma[ii,:,:,:],0)

                alpha_beta_gamma_i_up = tf.image.resize_bilinear(alpha_beta_gamma_i,(shapes_t[ii,0],shapes_t[ii,1]),align_corners=True) 

                alpha_hat = tf.expand_dims(alpha_beta_gamma_i_up[:,:,:,0],-1);
                gamma_hat = tf.expand_dims(alpha_beta_gamma_i_up[:,:,:,1],-1);
                beta_hat =  tf.expand_dims(alpha_beta_gamma_i_up[:,:,:,2],-1);

                alpha_ = tf.exp(alpha_hat)
                gamma_ = tf.exp(gamma_hat)
                beta_= (tf.exp((alpha_hat+gamma_hat)/2)*tf.tanh(beta_hat))

                a_list.append(alpha_)
                b_list.append(beta_)
                c_list.append(gamma_)

        return a_list,b_list,c_list
        
    

    
