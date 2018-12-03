"""
Created on Mon Apr 17 15:16:49 2017

@author: thanuja
"""
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import scipy as sp

#import matplotlib.cm as cm
import tensorflow as tf
h_o = 192
w_o = 640
mean = np.array([91.090, 95.435,  96.119], dtype=np.float32); 

def inv_preprocess(imgs, num_images):
  """Inverse preprocessing of the batch of images.
     Add the mean vector and convert from BGR to RGB.
   
  Args:
    imgs: batch of input images.
    num_images: number of images to apply the inverse transformations on.
  
  Returns:
    The batch of the size num_images with the same spatial dimensions as the input.
  """
  n, h, w, c = imgs.shape
  assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
  outputs = np.zeros((num_images, h_o, w_o, c), dtype=np.uint8)
  for i in range(num_images):
    im = sp.misc.imresize(imgs[i]+mean.astype(np.uint8),(h_o, w_o),interp='nearest')  
    outputs[i] = im
  return outputs


def prepare_depth(input_batch,num_images,inverse=False,vis_factor=60):
    """Resize masks and perform one-hot encoding.
    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    n,h,w,c = input_batch.shape


    outputs = np.zeros((num_images, h_o, w_o, 3), dtype=np.uint8)
    #pred = np.zeros((num_images, 128, 160), dtype=np.uint8)
    for i in range(num_images):
        #if h>128:
           if(inverse): 
               im = (np.uint8(plt.cm.jet(input_batch[i,:,:,0]/0.1)[:,:,0:3]*255))              
           else:
               im = (np.uint8(plt.cm.jet(input_batch[i,:,:,0]/vis_factor)[:,:,0:3]*255))
           im = sp.misc.imresize(im,(h_o, w_o),interp='nearest')
          

           outputs[i] = im   
    return outputs


     

def prepare_info(input_batch,num_images):
    """Resize masks and perform one-hot encoding.
    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    n,h,w,c = input_batch.shape
    #print np.sum(input_batch)
    #print(input_batch.shape) 

    outputs = np.zeros((num_images, h_o, w_o, 3), dtype=np.uint8)
    for i in range(num_images):
        
        factor = np.max(np.abs(input_batch[i,:,:,0]))*0.3
        im = (np.uint8(plt.cm.jet(np.abs(input_batch[i,:,:,0])/(factor))[:,:,0:3]*255)) 
        im = sp.misc.imresize(im,(h_o, w_o))
        
        outputs[i] = im   
    return outputs

def prepare_error(input_batch,num_images):
    """Resize masks and perform one-hot encoding.
    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    n,h,w,c = input_batch.shape
    #print np.sum(input_batch)
    #print(input_batch.shape) 
    logdepths_mean = 0#0.82473954
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        
        arr = input_batch[i,:,:,0]/(np.max(input_batch[i,:,:,0])+1e-8)
        arr = arr-np.min(arr)
        arr = arr/(np.max(input_batch[i,:,:,0])+1e-8)
        im = (np.uint8(plt.cm.jet(arr)[:,:,0:3]*255)) 
        outputs[i] = im   
    return outputs

def compute_maxes(input_batch,num_images):
    n,h,w,c = input_batch.shape

    maxes = np.zeros((num_images), dtype=np.float32)
    #pred = np.zeros((num_images, 128, 160), dtype=np.uint8)
    for i in range(0,n):
        maxes[i] = max(np.abs(input_batch[i,:,:,0:2].max()), np.abs(input_batch[i,:,:,0:2].min()))
    return maxes


def prepare_flow(input_batch,num_images,max_flow):
    """Resize masks and perform one-hot encoding.
    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    n,h,w,c = input_batch.shape


    output = np.zeros((num_images, h_o, w_o, 3), dtype=np.uint8)
   
    #pred = np.zeros((num_images, 128, 160), dtype=np.uint8)
    for i in range(0,n):
        flow_n = np.zeros([h,w,3],np.float64)
#        flow_n[:,:,0:2] = input_batch[i,:,:,:]/(max_flow*2) + 0.5
#            
#        flow_n[:,:,2] = 0.5*np.ones_like(np.squeeze(input_batch[0,:,:,0]))
#
#        flow_n = sp.misc.imresize(flow_n,(h_o,w_o),interp='nearest')
#        output[i] = flow_n*255
        
        du = input_batch[i,:,:,0];
        dv = input_batch[i,:,:,1];

        flow_n[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
        flow_n[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
        flow_n[:, :, 2] = abs(8 - flow_n[:, :, 1])
        
        small_idx = flow_n[:, :, 0:3] < 0
        large_idx = flow_n[:, :, 0:3] > 1
        
        flow_n[small_idx] = 0
        flow_n[large_idx] = 1
        
        # convert to rgb
        flow_n = cl.hsv_to_rgb(flow_n)
        flow_n = sp.misc.imresize(flow_n,(h_o,w_o),interp='nearest')
        
        output[i] = flow_n
    return output


