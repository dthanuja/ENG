"""
Created on Tue Aug 15 15:21:36 2017

@author: thanuja
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

c = 0.001

    
def vectorize_tensor(tensor):
    """Vectorizes an input tensor
    Args:
      tensor: of the form [batch, channels, height_s, width_s] or [batch, channels, length]
    Returns:
      vectorized tensor [batch, channels, height_t*width_t] or [batch, channels, length]
    """
    shape = tf.shape(tensor)#tensor.get_shape().as_list()
    #print(len(shape.shape))
#    if len(shape.shape) == 4: # batch included in tensor
    return tf.reshape(tensor, [shape[0], shape[1], -1])

def compute_pose(depths,flow,c_x,c_y,c_xy,batchsize,k):
    
     
     #shape = depths.get_shape().as_list()
     shape = tf.shape(depths)
     
     img_xy = getmeshgrid(batchsize,height=shape[1],width=shape[2])
     
     img_xy = tf.transpose(img_xy,perm=[0,2,3,1])
#     outlist.append(img_xy)
     img_xy1 = tf.concat((img_xy, tf.expand_dims(tf.ones_like(img_xy[:,:,:,0]),-1)),3)

     xyz1 = unproject_2(depths, img_xy1, k, batchsize)
#     outlist.append(xyz1)
     
     
     #print flow.shape
     flow_r = tf.reshape(flow,[batchsize,shape[1]*shape[2],2])
     #print flow_r.shape
     flow_r = tf.transpose(flow_r,[0,2,1])

     flow_x = flow_r[:,0,:]/tf.expand_dims(k[:,0,0],-1)
     flow_y = flow_r[:,1,:]/tf.expand_dims(k[:,1,1],-1)

     error = tf.concat([flow_x,flow_y],-1)

     
     error = tf.expand_dims(error,1)
     #print error.shape
#     
#     
#     outlist.append(error)
     J1,J2,J = jac_flow(xyz1)

     c_x = tf.reshape(c_x,[batchsize,-1,1])
     c_y = tf.reshape(c_y,[batchsize,-1,1])
     c_xy = tf.reshape(c_xy,[batchsize,-1,1])

     c_x_r = tf.transpose(c_x,[0,2,1])
     c_y_r = tf.transpose(c_y,[0,2,1])
     c_xy_r = tf.transpose(c_xy,[0,2,1])
     
     c_x_r = c_x_r[:,0,:]
     c_y_r = c_y_r[:,0,:]
     c_xy_r = c_xy_r[:,0,:]
     

     c_x_J10C = c_x_r*J1[:,0,:]
     c_x_J11C = c_x_r*J1[:,1,:]
     c_x_J12C = c_x_r*J1[:,2,:]
     c_x_J13C = c_x_r*J1[:,3,:]
     c_x_J14C = c_x_r*J1[:,4,:]
     c_x_J15C = c_x_r*J1[:,5,:]
     
     
     c_xy_J20C = c_xy_r*J2[:,0,:]
     c_xy_J21C = c_xy_r*J2[:,1,:]
     c_xy_J22C = c_xy_r*J2[:,2,:]
     c_xy_J23C = c_xy_r*J2[:,3,:]
     c_xy_J24C = c_xy_r*J2[:,4,:]
     c_xy_J25C = c_xy_r*J2[:,5,:]
     
     
     c_xy_J10C = c_xy_r*J1[:,0,:]
     c_xy_J11C = c_xy_r*J1[:,1,:]
     c_xy_J12C = c_xy_r*J1[:,2,:]
     c_xy_J13C = c_xy_r*J1[:,3,:]
     c_xy_J14C = c_xy_r*J1[:,4,:]
     c_xy_J15C = c_xy_r*J1[:,5,:]
     
     c_y_J20C = c_y_r*J2[:,0,:]
     c_y_J21C = c_y_r*J2[:,1,:]
     c_y_J22C = c_y_r*J2[:,2,:]
     c_y_J23C = c_y_r*J2[:,3,:]
     c_y_J24C = c_y_r*J2[:,4,:]
     c_y_J25C = c_y_r*J2[:,5,:]
    
     
     col_0_0 = c_x_J10C+c_xy_J20C
     col_0_1 = c_x_J11C+c_xy_J21C
     col_0_2 = c_x_J12C+c_xy_J22C
     col_0_3 = c_x_J13C+c_xy_J23C
     col_0_4 = c_x_J14C+c_xy_J24C
     col_0_5 = c_x_J15C+c_xy_J25C
     
     col_1_0 = c_xy_J10C+c_y_J20C
     col_1_1 = c_xy_J11C+c_y_J21C
     col_1_2 = c_xy_J12C+c_y_J22C
     col_1_3 = c_xy_J13C+c_y_J23C
     col_1_4 = c_xy_J14C+c_y_J24C
     col_1_5 = c_xy_J15C+c_y_J25C
     
     Sigma_J_1 = tf.stack([col_0_0, col_0_1, col_0_2, col_0_3, col_0_4, col_0_5],1)
     Sigma_J_2 = tf.stack([col_1_0, col_1_1, col_1_2, col_1_3, col_1_4, col_1_5],1)
     
     Sigma_J = tf.concat([Sigma_J_1,Sigma_J_2],-1)  
     
     

     
     Sigma_J_t = tf.transpose(Sigma_J,perm=[0,2,1])
     
     # initialise cholesky with synmetric matrix J^TWJ
     J_t_Sigma_j = tf.matmul(J,Sigma_J_t) + 0.0001*tf.eye(6,batch_shape=[batchsize])
     J_t_e = tf.matmul(error,Sigma_J_t)

     chol =  tf.cholesky(J_t_Sigma_j)
     
     # solve for update delta = (J^T.W.J)^-1 . (J^T.W*error) via cholesky
     delta = tf.cholesky_solve(chol,tf.transpose(J_t_e,[0,2,1]))
     
     return delta



        
def jac_flow(xyz1):
    
    xyz1_vec =vectorize_tensor(xyz1) #[batch,channels,h*w]
    mask = tf.greater(tf.expand_dims(xyz1_vec[:,2,:],1), 0)
    mask = tf.concat((mask,mask,mask,mask),1)


    uv1q_vec = xyz1_vec/tf.expand_dims(xyz1_vec[:,2,:],1) 
    uv1q_vec = tf.where(mask,uv1q_vec,tf.zeros_like(uv1q_vec))   
    

    
    q = uv1q_vec[:,-1,:] 
    u = uv1q_vec[:,0,:]
    v = uv1q_vec[:,1,:] 
    zeros = tf.zeros_like(v)
    
    J1 = tf.stack([q, zeros, -u*q, -u*v, u*u+1, -v],1)
    J2 = tf.stack([zeros, q ,-v*q , -v*v-1, u*v ,u],1)

    J = tf.concat([J1,J2],-1)                      
    return J1,J2,J
    
    
# modified to be just x,y
def getmeshgrid(batch_size,height,width):
    """Extends tf.meshgrid to operate across batches
      Args:
        batch: batch size
        height: height of meshgrid
        width: width of the meshgrid
      Returns:
        meshgrid of homogeneous  pixel coords [batch, channels=3, height, width] 
          as float32
    """
    xx,yy = tf.meshgrid(tf.range(0,width),tf.range(0,height))
#    zz = tf.ones_like(xx);
    xy = tf.stack([xx,yy]);
    xy = tf.expand_dims(xy, 0)
    grid = tf.tile(xy, [batch_size, 1,1,1])   
    grid = tf.cast(grid,tf.float32)
    return grid

#
def unproject_2(d, uv1, K, batch):
    """Computes the homogeneous camera coords, given a set of pixel coords, 
      and corresponding depths
    Args:
      d: depth image [batch, height, width]
      uv1: homogeneous image/pixel coordinates [batch, channels=3, height, width]
      K: camera matrix [batch, 3, 3]
      batch: batch size
    Returns:
      homogeneous camera coords [batch, channels=4, height_t, width_t]
    """
    #shape = d.get_shape().as_list()
    shape = tf.shape(d)
#    print('unporject 2')
#    print(shape)
    
    
    uv1 = tf.transpose(uv1, perm=[0, 3, 1, 2])
    #batch, height, width = d.get_shape().as_list()
    d = tf.reshape(d,[batch,1,-1])
    xyz = tf.matmul(tf.matrix_inverse(K), tf.reshape(uv1,[batch,3,-1])) * d
    

    # set homogeneous coord to zero if invalid
    ones = tf.where(tf.greater(d,0.1), tf.ones_like(d),tf.zeros_like(d))
    
   
    xyz1 = tf.concat([xyz, ones], axis=1)
    xyz1 = tf.reshape(xyz1, [batch, -1, shape[1], shape[2]])
    
    return xyz1
   
def project_w_K(xyz1, K, height, width):
    """Projects the homogeneous world coordinates onto pixel coords, used for 
      correspondences
    Args:
      xyz1: homogeneous coords as a vector [batch, channels=4, length], or [batch, 
        channels=4, height, width] height/width correspond to the dimensions of the 
        output image, length = height*width. The 4 channels correspond to the 
        homogeneous coordinates
      K: camera matrix [batch, 3, 3].  
      height: output size, if vectorized already, this must be input to function 
      width: output size, if vectorized already, this must be input to function 
    Returns:
      homogeneous pixel coords [batch, height_t, width_t, channels=3]
    """
    # get the shape of the input tensor
    shape = tf.shape(xyz1)

    batch = shape[0]
    xyz1 = vectorize_tensor(xyz1)
    

    xyz_coords = xyz1[:,0:3,:]
    
    projection = tf.matmul(K,xyz_coords);
    # computed normalised cam coords
    x_u = projection[:,0,:] # lambda.u
    y_u = projection[:,1,:] # lambda.v
    z_u = projection[:,2,:] # lambda
    
    
    # compute pixel coords
    u = x_u/z_u;
    v = y_u/z_u;
    
    # mask for invalid depths
    mask = tf.greater(z_u, 0.1)
#    print("mask")
#    nans = tf.reduce_sum(mask)

    
#    u = u*mask
    u_2 = tf.where(mask,u,tf.zeros_like(u))   
    v_2 = tf.where(mask,v,tf.zeros_like(v))   

   
    # homogeneous coordinates
    pixel_coords = tf.concat([u_2, v_2, tf.cast(mask, tf.float32)], axis=1)
    
    # reshape to image shape for bilinear filter
    pixel_coords = tf.reshape(pixel_coords, [batch, 3, height, width])
    pixel_coords =  tf.transpose(pixel_coords, perm=[0, 2, 3, 1])
    
    # remove invalid values
#    nans = tf.reduce_sum(tf.cast(tf.is_nan(pixel_coords),tf.float32))

    return pixel_coords


def expSE3(x,batch):

    one_6th = tf.constant(1.0/6.0);
    one_20th = tf.constant(1.0/20.0);
    
    
    w = x[:,3:6];

    theta_sq = tf.tensordot(w,w,axes=[[1],[1]])
    #theta_sq = tf.matmul(w,w,transpose_a=False,transpose_b=True)
    theta_sq = theta_sq[:,0]

    theta = tf.sqrt(theta_sq);
    #w = tf.squeeze(w) 
    cross_ = tf.cross(w,x[:,0:3])
    
    
    A1 = 1.0 - one_6th * theta_sq;
    B1 = 0.5*tf.ones_like(A1);
    translation1 = (x[:,0:3] + 0.5 * cross_);
     
    C2 = one_6th*(1.0 - one_20th * theta_sq);
    A2 = 1.0 - theta_sq * C2;
    B2 = 0.5 - 0.25 * one_6th * theta_sq;
     
    inv_theta = 1.0/theta;
    A3 = tf.sin(theta) * inv_theta;
    B3 = (1 - tf.cos(theta)) * (inv_theta * inv_theta);
    C3 = (1 - A3) * (inv_theta * inv_theta);
  
    
    bool_tensor = tf.less(theta_sq,1e-8)
    
    bool_tensor2 = tf.less(theta_sq,1e-6)
    A_Alternate = tf.where(bool_tensor2,A2,A3)
    B_Alternate = tf.where(bool_tensor2,B2,B3) 
    C = tf.where(bool_tensor2,C2,C3) 
    
    
    A = tf.where(bool_tensor,A1,A_Alternate)
    B = tf.where(bool_tensor,B1,B_Alternate)

    translation2 = (x[:,0:3] + tf.expand_dims(B,-1)*cross_ + tf.expand_dims(C,-1)*(tf.cross(w, cross_)));
    
    translation = tf.where(bool_tensor,translation1,translation2)
    
         
       
    wx2 = w[:,0]*w[:,0]
    wy2 = w[:,1]*w[:,1]
    wz2 = w[:,2]*w[:,2]
    
    r00 = 1.0 -  B*(wy2+wz2)
    r11 = 1.0 -  B*(wx2+wz2)
    r22 = 1.0 -  B*(wx2+wz2)
    
    a = A*w[:,2]
    b = B*(w[:,0]*w[:,1])
    r01 = b-a;
    r10 = b+a;
    
    a = A*w[:,1]
    b = B*(w[:,0]*w[:,2])
    r02 = b+a;
    r20 = b-a;
    
    a = A*w[:,0]
    b = B*(w[:,1]*w[:,2])
    r12 = b-a
    r21 = b+a
    
    c0 = tf.stack([r00,r01,r02,translation[:,0]]);
    c1 = tf.stack([r10,r11,r12,translation[:,1]]);
    c2 = tf.stack([r20,r21,r22,translation[:,2]]);
    
    c4 = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1,4])
    c4 = tf.transpose(tf.tile(c4, [batch,1 ]))
  
    SE3out =tf.transpose(tf.stack([c0,c1,c2,c4]),[2,0,1])
    
       
    return SE3out        
        


