"""
Created on Thu Apr  6 13:59:32 2017

@author: thanuja
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt


IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

class DataSetNew:
    def __init__(self, batch_size,mode='Train',gen_log=False):
        self.batch_size = batch_size
        self.mode = mode
        self.gen_log = gen_log
        #self.batch_size_tensor = tf.placeholder_with_default(batch_size, shape=(None))
        self.batch_size_tensor = batch_size
        self.mean = tf.constant([91.090, 95.435,  96.119], dtype=tf.float32);
        
    def create_img(self, tf_png, height, width, channel_cnt, tf_type_in, tf_type_out):
        img = tf.image.decode_png(tf_png, channel_cnt, tf_type_in)     
        img = tf.cast(img, tf_type_out)
        
        img.set_shape((height,width,channel_cnt))
        
        return img
    
    def csv_inputs(self, csv_file_path):
        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=False)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        rgb_file1,rgb_file2,height_,width_,fx,cx,fy,cy= tf.decode_csv(serialized_example, [["rgb1"],["rgb2"],['height'],['width'],['fx'],['cx'],['fy'],['cy']])
      

        k_vals = tf.string_to_number([fx,cx,fy,cy],tf.float32)

        k = [[k_vals[0],0.0,k_vals[1]],[0,k_vals[2],k_vals[3]],[0.0,0.0,1.0]]
        intrinsics = tf.convert_to_tensor(k,dtype=tf.float32)
        
        tfpng_1 = tf.read_file(rgb_file1)
        tfpng_2 = tf.read_file(rgb_file2)
        

        
        TARGET_HEIGHT = tf.string_to_number(height_,tf.int32)
        TARGET_WIDTH = tf.string_to_number(width_,tf.int32)
        shape_t = tf.stack([TARGET_HEIGHT,TARGET_WIDTH],0)

        
      
        image_1 = self.create_img(tfpng_1, IMAGE_HEIGHT, IMAGE_WIDTH, 3, tf.uint8, tf.float32)
        image_2 = self.create_img(tfpng_2, IMAGE_HEIGHT, IMAGE_WIDTH, 3, tf.uint8, tf.float32)
        
        image_1 = image_1-self.mean;  
        image_2 = image_2-self.mean;
        
        img_stack = tf.concat([image_1, image_2], 2)

       
        img_stacks,cams,shapes= tf.train.batch(
            [img_stack,intrinsics,shape_t],
            batch_size=self.batch_size_tensor,
            num_threads=1,
            capacity= 3
        )
        
        return  img_stacks,cams,shapes

