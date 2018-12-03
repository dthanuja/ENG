#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:05:11 2017

@author: thanuja
"""

from .network import Network
import tensorflow as tf
import numpy as np

class depthflowPoseNet(Network):
     def setup(self,is_training,keep_prob_=1.0):
        # input structural information at each level too
        (self.feed('data_')
             .conv(7, 7, 96, 2, 2,relu=False, name='conv1')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv1_bn')
             .max_pool(3, 3, 2, 2, name='pool1')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_1_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv2_1_x1')
             .dropout(name = 'conv2_1_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_1_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv2_1_x2'))
#
        (self.feed('pool1', 
                   'conv2_1_x2')
             .concat(3, name='concat_2_1')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_2_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv2_2_x1')
             .dropout(name = 'conv2_2_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_2_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv2_2_x2'))

        (self.feed('concat_2_1', 
                   'conv2_2_x2')
             .concat(3, name='concat_2_2')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_3_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv2_3_x1')
             .dropout(name = 'conv2_3_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_3_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv2_3_x2'))

        (self.feed('concat_2_2', 
                   'conv2_3_x2')
             .concat(3, name='concat_2_3')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_4_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv2_4_x1')
             .dropout(name = 'conv2_4_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_4_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv2_4_x2'))

        (self.feed('concat_2_3', 
                   'conv2_4_x2')
             .concat(3, name='concat_2_4')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_5_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv2_5_x1')
             .dropout(name = 'conv2_5_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_5_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv2_5_x2'))

        (self.feed('concat_2_4', 
                   'conv2_5_x2')
             .concat(3, name='concat_2_5')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_6_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv2_6_x1')
             .dropout(name = 'conv2_6_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_6_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv2_6_x2'))

        (self.feed('concat_2_5', 
                   'conv2_6_x2')
             .concat(3, name='concat_2_6')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv2_blk_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv2_blk')
             .dropout(name = 'conv2_blk_drop', keep_prob=keep_prob_)
             .avg_pool(3, 3, 2, 2, name='pool2')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_1_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3_1_x1')
             .dropout(name = 'conv3_1_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_1_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3_1_x2'))

        (self.feed('pool2', 
                   'conv3_1_x2')
             .concat(3, name='concat_3_1')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_2_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3_2_x1')
             .dropout(name = 'conv3_2_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_2_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3_2_x2'))

        (self.feed('concat_3_1', 
                   'conv3_2_x2')
             .concat(3, name='concat_3_2')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_3_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3_3_x1')
             .dropout(name = 'conv3_3_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_3_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3_3_x2'))

        (self.feed('concat_3_2', 
                   'conv3_3_x2')
             .concat(3, name='concat_3_3')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_4_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3_4_x1')
             .dropout(name = 'conv3_4_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_4_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3_4_x2'))

        (self.feed('concat_3_3', 
                   'conv3_4_x2')
             .concat(3, name='concat_3_4')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_5_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3_5_x1')
             .dropout(name = 'conv3_5_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_5_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3_5_x2'))

        (self.feed('concat_3_4', 
                   'conv3_5_x2')
             .concat(3, name='concat_3_5')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_6_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3_6_x1')
             .dropout(name = 'conv3_6_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_6_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3_6_x2'))

        (self.feed('concat_3_5', 
                   'conv3_6_x2')
             .concat(3, name='concat_3_6')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_7_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3_7_x1')
             .dropout(name = 'conv3_7_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_7_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3_7_x2'))

        (self.feed('concat_3_6', 
                   'conv3_7_x2')
             .concat(3, name='concat_3_7')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_8_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3_8_x1')
             .dropout(name = 'conv3_8_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_8_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3_8_x2'))

        (self.feed('concat_3_7', 
                   'conv3_8_x2')
             .concat(3, name='concat_3_8')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_9_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3_9_x1')
             .dropout(name = 'conv3_9_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_9_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3_9_x2'))

        (self.feed('concat_3_8', 
                   'conv3_9_x2')
             .concat(3, name='concat_3_9')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_10_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3_10_x1')
             .dropout(name = 'conv3_10_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_10_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3_10_x2'))

        (self.feed('concat_3_9', 
                   'conv3_10_x2')
             .concat(3, name='concat_3_10')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_11_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3_11_x1')
             .dropout(name = 'conv3_11_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_11_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3_11_x2'))

        (self.feed('concat_3_10', 
                   'conv3_11_x2')
             .concat(3, name='concat_3_11')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_12_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3_12_x1')
             .dropout(name = 'conv3_12_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_12_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3_12_x2'))

        (self.feed('concat_3_11', 
                   'conv3_12_x2')
             .concat(3, name='concat_3_12')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv3_blk_bn')
             .conv(1, 1, 384, 1, 1, biased=False, relu=False, name='conv3_blk')
             .dropout(name = 'conv3_blk_drop', keep_prob=keep_prob_)
             .avg_pool(3, 3, 2, 2, name='pool3')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_1_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_1_x1')
             .dropout(name = 'conv4_1_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_1_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_1_x2'))

        (self.feed('pool3', 
                   'conv4_1_x2')
             .concat(3, name='concat_4_1')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_2_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_2_x1')
             .dropout(name = 'conv4_2_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_2_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_2_x2'))

        (self.feed('concat_4_1', 
                   'conv4_2_x2')
             .concat(3, name='concat_4_2')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_3_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_3_x1')
             .dropout(name = 'conv4_3_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_3_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_3_x2'))

        (self.feed('concat_4_2', 
                   'conv4_3_x2')
             .concat(3, name='concat_4_3')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_4_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_4_x1')
             .dropout(name = 'conv4_4_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_4_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_4_x2'))

        (self.feed('concat_4_3', 
                   'conv4_4_x2')
             .concat(3, name='concat_4_4')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_5_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_5_x1')
             .dropout(name = 'conv4_5_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_5_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_5_x2'))

        (self.feed('concat_4_4', 
                   'conv4_5_x2')
             .concat(3, name='concat_4_5')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_6_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_6_x1')
             .dropout(name = 'conv4_6_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_6_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_6_x2'))

        (self.feed('concat_4_5', 
                   'conv4_6_x2')
             .concat(3, name='concat_4_6')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_7_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_7_x1')
             .dropout(name = 'conv4_7_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_7_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_7_x2'))

        (self.feed('concat_4_6', 
                   'conv4_7_x2')
             .concat(3, name='concat_4_7')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_8_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_8_x1')
             .dropout(name = 'conv4_8_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_8_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_8_x2'))

        (self.feed('concat_4_7', 
                   'conv4_8_x2')
             .concat(3, name='concat_4_8')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_9_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_9_x1')
             .dropout(name = 'conv4_9_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_9_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_9_x2'))

        (self.feed('concat_4_8', 
                   'conv4_9_x2')
             .concat(3, name='concat_4_9')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_10_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_10_x1')
             .dropout(name = 'conv4_10_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_10_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_10_x2'))

        (self.feed('concat_4_9', 
                   'conv4_10_x2')
             .concat(3, name='concat_4_10')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_11_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_11_x1')
             .dropout(name = 'conv4_11_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_11_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_11_x2'))

        (self.feed('concat_4_10', 
                   'conv4_11_x2')
             .concat(3, name='concat_4_11')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_12_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_12_x1')
             .dropout(name = 'conv4_12_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_12_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_12_x2'))

        (self.feed('concat_4_11', 
                   'conv4_12_x2')
             .concat(3, name='concat_4_12')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_13_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_13_x1')
             .dropout(name = 'conv4_13_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_13_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_13_x2'))

        (self.feed('concat_4_12', 
                   'conv4_13_x2')
             .concat(3, name='concat_4_13')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_14_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_14_x1')
             .dropout(name = 'conv4_14_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_14_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_14_x2'))

        (self.feed('concat_4_13', 
                   'conv4_14_x2')
             .concat(3, name='concat_4_14')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_15_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_15_x1')
             .dropout(name = 'conv4_15_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_15_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_15_x2'))

        (self.feed('concat_4_14', 
                   'conv4_15_x2')
             .concat(3, name='concat_4_15')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_16_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_16_x1')
             .dropout(name = 'conv4_16_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_16_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_16_x2'))

        (self.feed('concat_4_15', 
                   'conv4_16_x2')
             .concat(3, name='concat_4_16')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_17_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_17_x1')
             .dropout(name = 'conv4_17_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_17_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_17_x2'))

        (self.feed('concat_4_16', 
                   'conv4_17_x2')
             .concat(3, name='concat_4_17')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_18_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_18_x1')
             .dropout(name = 'conv4_18_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_18_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_18_x2'))

        (self.feed('concat_4_17', 
                   'conv4_18_x2')
             .concat(3, name='concat_4_18')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_19_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_19_x1')
             .dropout(name = 'conv4_19_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_19_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_19_x2'))

        (self.feed('concat_4_18', 
                   'conv4_19_x2')
             .concat(3, name='concat_4_19')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_20_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_20_x1')
             .dropout(name = 'conv4_20_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_20_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_20_x2'))

        (self.feed('concat_4_19', 
                   'conv4_20_x2')
             .concat(3, name='concat_4_20')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_21_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_21_x1')
             .dropout(name = 'conv4_21_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_21_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_21_x2'))

        (self.feed('concat_4_20', 
                   'conv4_21_x2')
             .concat(3, name='concat_4_21')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_22_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_22_x1')
             .dropout(name = 'conv4_22_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_22_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_22_x2'))

        (self.feed('concat_4_21', 
                   'conv4_22_x2')
             .concat(3, name='concat_4_22')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_23_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_23_x1')
             .dropout(name = 'conv4_23_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_23_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_23_x2'))

        (self.feed('concat_4_22', 
                   'conv4_23_x2')
             .concat(3, name='concat_4_23')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_24_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_24_x1')
             .dropout(name = 'conv4_24_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_24_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_24_x2'))

        (self.feed('concat_4_23', 
                   'conv4_24_x2')
             .concat(3, name='concat_4_24')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_25_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_25_x1')
             .dropout(name = 'conv4_25_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_25_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_25_x2'))

        (self.feed('concat_4_24', 
                   'conv4_25_x2')
             .concat(3, name='concat_4_25')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_26_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_26_x1')
             .dropout(name = 'conv4_26_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_26_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_26_x2'))

        (self.feed('concat_4_25', 
                   'conv4_26_x2')
             .concat(3, name='concat_4_26')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_27_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_27_x1')
             .dropout(name = 'conv4_27_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_27_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_27_x2'))

        (self.feed('concat_4_26', 
                   'conv4_27_x2')
             .concat(3, name='concat_4_27')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_28_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_28_x1')
             .dropout(name = 'conv4_28_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_28_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_28_x2'))

        (self.feed('concat_4_27', 
                   'conv4_28_x2')
             .concat(3, name='concat_4_28')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_29_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_29_x1')
             .dropout(name = 'conv4_29_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_29_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_29_x2'))

        (self.feed('concat_4_28', 
                   'conv4_29_x2')
             .concat(3, name='concat_4_29')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_30_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_30_x1')
             .dropout(name = 'conv4_30_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_30_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_30_x2'))

        (self.feed('concat_4_29', 
                   'conv4_30_x2')
             .concat(3, name='concat_4_30')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_31_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_31_x1')
             .dropout(name = 'conv4_31_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_31_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_31_x2'))

        (self.feed('concat_4_30', 
                   'conv4_31_x2')
             .concat(3, name='concat_4_31')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_32_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_32_x1')
             .dropout(name = 'conv4_32_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_32_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_32_x2'))

        (self.feed('concat_4_31', 
                   'conv4_32_x2')
             .concat(3, name='concat_4_32')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_33_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_33_x1')
             .dropout(name = 'conv4_33_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_33_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_33_x2'))

        (self.feed('concat_4_32', 
                   'conv4_33_x2')
             .concat(3, name='concat_4_33')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_34_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_34_x1')
             .dropout(name = 'conv4_34_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_34_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_34_x2'))

        (self.feed('concat_4_33', 
                   'conv4_34_x2')
             .concat(3, name='concat_4_34')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_35_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_35_x1')
             .dropout(name = 'conv4_35_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_35_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_35_x2'))

        (self.feed('concat_4_34', 
                   'conv4_35_x2')
             .concat(3, name='concat_4_35')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_36_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4_36_x1')
             .dropout(name = 'conv4_36_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_36_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4_36_x2'))

        (self.feed('concat_4_35', 
                   'conv4_36_x2')
             .concat(3, name='concat_4_36')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv4_blk_bn')
             .conv(1, 1, 1056, 1, 1, biased=False, relu=False, name='conv4_blk')
             .dropout(name = 'conv4_blk_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_1_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_1_x1')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_1_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_1_x2'))

        (self.feed('conv4_blk', 
                   'conv5_1_x2')
             .concat(3, name='concat_5_1')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_2_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_2_x1')
             .dropout(name = 'conv5_2_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_2_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_2_x2'))

        (self.feed('concat_5_1', 
                   'conv5_2_x2')
             .concat(3, name='concat_5_2')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_3_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_3_x1')
             .dropout(name = 'conv5_3_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_3_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_3_x2'))

        (self.feed('concat_5_2', 
                   'conv5_3_x2')
             .concat(3, name='concat_5_3')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_4_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_4_x1')
             .dropout(name = 'conv5_4_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_4_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_4_x2'))

        (self.feed('concat_5_3', 
                   'conv5_4_x2')
             .concat(3, name='concat_5_4')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_5_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_5_x1')
             .dropout(name = 'conv5_5_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_5_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_5_x2'))

        (self.feed('concat_5_4', 
                   'conv5_5_x2')
             .concat(3, name='concat_5_5')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_6_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_6_x1')
             .dropout(name = 'conv5_6_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_6_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_6_x2'))

        (self.feed('concat_5_5', 
                   'conv5_6_x2')
             .concat(3, name='concat_5_6')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_7_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_7_x1')
             .dropout(name = 'conv5_7_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_7_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_7_x2'))

        (self.feed('concat_5_6', 
                   'conv5_7_x2')
             .concat(3, name='concat_5_7')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_8_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_8_x1')
             .dropout(name = 'conv5_8_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_8_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_8_x2'))

        (self.feed('concat_5_7', 
                   'conv5_8_x2')
             .concat(3, name='concat_5_8')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_9_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_9_x1')
             .dropout(name = 'conv5_9_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_9_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_9_x2'))

        (self.feed('concat_5_8', 
                   'conv5_9_x2')
             .concat(3, name='concat_5_9')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_10_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_10_x1')
             .dropout(name = 'conv5_10_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_10_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_10_x2'))

        (self.feed('concat_5_9', 
                   'conv5_10_x2')
             .concat(3, name='concat_5_10')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_11_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_11_x1')
             .dropout(name = 'conv5_11_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_11_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_11_x2'))

        (self.feed('concat_5_10', 
                   'conv5_11_x2')
             .concat(3, name='concat_5_11')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_12_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_12_x1')
             .dropout(name = 'conv5_12_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_12_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_12_x2'))

        (self.feed('concat_5_11', 
                   'conv5_12_x2')
             .concat(3, name='concat_5_12')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_13_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_13_x1')
             .dropout(name = 'conv5_13_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_13_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_13_x2'))

        (self.feed('concat_5_12', 
                   'conv5_13_x2')
             .concat(3, name='concat_5_13')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_14_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_14_x1')
             .dropout(name = 'conv5_14_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_14_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_14_x2'))

        (self.feed('concat_5_13', 
                   'conv5_14_x2')
             .concat(3, name='concat_5_14')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_15_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_15_x1')
             .dropout(name = 'conv5_15_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_15_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_15_x2'))

        (self.feed('concat_5_14', 
                   'conv5_15_x2')
             .concat(3, name='concat_5_15')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_16_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_16_x1')
             .dropout(name = 'conv5_16_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_16_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_16_x2'))

        (self.feed('concat_5_15', 
                   'conv5_16_x2')
             .concat(3, name='concat_5_16')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_17_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_17_x1')
             .dropout(name = 'conv5_17_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_17_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_17_x2'))

        (self.feed('concat_5_16', 
                   'conv5_17_x2')
             .concat(3, name='concat_5_17')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_18_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_18_x1')
             .dropout(name = 'conv5_18_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_18_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_18_x2'))

        (self.feed('concat_5_17', 
                   'conv5_18_x2')
             .concat(3, name='concat_5_18')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_19_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_19_x1')
             .dropout(name = 'conv5_19_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_19_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_19_x2'))

        (self.feed('concat_5_18', 
                   'conv5_19_x2')
             .concat(3, name='concat_5_19')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_20_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_20_x1')
             .dropout(name = 'conv5_20_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_20_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_20_x2'))

        (self.feed('concat_5_19', 
                   'conv5_20_x2')
             .concat(3, name='concat_5_20')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_21_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_21_x1')
             .dropout(name = 'conv5_21_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_21_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_21_x2'))

        (self.feed('concat_5_20', 
                   'conv5_21_x2')
             .concat(3, name='concat_5_21')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_22_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_22_x1')
             .dropout(name = 'conv5_22_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_22_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_22_x2'))

        (self.feed('concat_5_21', 
                   'conv5_22_x2')
             .concat(3, name='concat_5_22')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_23_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_23_x1')
             .dropout(name = 'conv5_23_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_23_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_23_x2'))

        (self.feed('concat_5_22', 
                   'conv5_23_x2')
             .concat(3, name='concat_5_23')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_24_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5_24_x1')
             .dropout(name = 'conv5_24_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_24_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5_24_x2'))

        (self.feed('concat_5_23', 
                   'conv5_24_x2')
             .concat(3, name='concat_5_24')
             .batch_normalization(is_training=False, activation_fn=tf.nn.relu, name='conv5_blk_bn')
      	     .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='layer1')
             .dropout(name = 'conv5_24_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False, activation_fn=None, name='layer1_BN')             
             .up_project([3, 3, 1024, 512], id = '2x', stride = 1, BN=True))
        
        up2 = self.terminals[-1]
        (self.feed('conv3_blk', 
                   up2)
             .concat(3, name='up2')
             .up_project([3, 3, 896, 584], id = '8x', stride = 1, BN=True))
        up3 = self.terminals[-1]
        (self.feed('conv2_blk', 
                   up3)
             .concat(3, name='up3')
             .up_project([3, 3, 776, 256], id = '16x', stride = 1, BN=True))
        up4 = self.terminals[-1]
        
        
        (self.feed(up4)
             .up_project([3, 3, 256, 128], id = '32x', stride = 1, BN=True))
        
        terminal = self.terminals[-1]
        
        print(np.shape(terminal))
        (self.feed(terminal)
             .conv(3, 3, 1, 1, 1, name = 'ConvPred'))
        
        depth_pred1 = self.terminals[-1]
        self.depth_preds.append(depth_pred1)
        
        (self.feed('data1_')
             .conv(7, 7, 96, 2, 2,relu=False,reuse=True, name='conv1')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv1_bn')
             .max_pool(3, 3, 2, 2, name='pool1')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_1_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv2_1_x1')
             .dropout(name = 'conv2_1_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_1_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv2_1_x2'))
#
        (self.feed('pool1', 
                   'conv2_1_x2')
             .concat(3, name='concat_2_1')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_2_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv2_2_x1')
             .dropout(name = 'conv2_2_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_2_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv2_2_x2'))

        (self.feed('concat_2_1', 
                   'conv2_2_x2')
             .concat(3, name='concat_2_2')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_3_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv2_3_x1')
             .dropout(name = 'conv2_3_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_3_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv2_3_x2'))

        (self.feed('concat_2_2', 
                   'conv2_3_x2')
             .concat(3, name='concat_2_3')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_4_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv2_4_x1')
             .dropout(name = 'conv2_4_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_4_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv2_4_x2'))

        (self.feed('concat_2_3', 
                   'conv2_4_x2')
             .concat(3, name='concat_2_4')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_5_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv2_5_x1')
             .dropout(name = 'conv2_5_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_5_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv2_5_x2'))

        (self.feed('concat_2_4', 
                   'conv2_5_x2')
             .concat(3, name='concat_2_5')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_6_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv2_6_x1')
             .dropout(name = 'conv2_6_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_6_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv2_6_x2'))

        (self.feed('concat_2_5', 
                   'conv2_6_x2')
             .concat(3, name='concat_2_6')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv2_blk_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv2_blk')
             .dropout(name = 'conv2_blk_drop', keep_prob=keep_prob_)
             .avg_pool(3, 3, 2, 2, name='pool2')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_1_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv3_1_x1')
             .dropout(name = 'conv3_1_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_1_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv3_1_x2'))

        (self.feed('pool2', 
                   'conv3_1_x2')
             .concat(3, name='concat_3_1')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_2_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv3_2_x1')
             .dropout(name = 'conv3_2_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_2_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv3_2_x2'))

        (self.feed('concat_3_1', 
                   'conv3_2_x2')
             .concat(3, name='concat_3_2')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_3_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv3_3_x1')
             .dropout(name = 'conv3_3_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_3_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv3_3_x2'))

        (self.feed('concat_3_2', 
                   'conv3_3_x2')
             .concat(3, name='concat_3_3')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_4_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv3_4_x1')
             .dropout(name = 'conv3_4_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_4_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv3_4_x2'))

        (self.feed('concat_3_3', 
                   'conv3_4_x2')
             .concat(3, name='concat_3_4')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_5_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv3_5_x1')
             .dropout(name = 'conv3_5_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_5_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv3_5_x2'))

        (self.feed('concat_3_4', 
                   'conv3_5_x2')
             .concat(3, name='concat_3_5')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_6_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv3_6_x1')
             .dropout(name = 'conv3_6_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_6_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv3_6_x2'))

        (self.feed('concat_3_5', 
                   'conv3_6_x2')
             .concat(3, name='concat_3_6')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_7_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv3_7_x1')
             .dropout(name = 'conv3_7_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_7_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv3_7_x2'))

        (self.feed('concat_3_6', 
                   'conv3_7_x2')
             .concat(3, name='concat_3_7')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_8_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv3_8_x1')
             .dropout(name = 'conv3_8_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_8_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv3_8_x2'))

        (self.feed('concat_3_7', 
                   'conv3_8_x2')
             .concat(3, name='concat_3_8')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_9_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv3_9_x1')
             .dropout(name = 'conv3_9_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_9_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv3_9_x2'))

        (self.feed('concat_3_8', 
                   'conv3_9_x2')
             .concat(3, name='concat_3_9')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_10_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv3_10_x1')
             .dropout(name = 'conv3_10_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_10_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv3_10_x2'))

        (self.feed('concat_3_9', 
                   'conv3_10_x2')
             .concat(3, name='concat_3_10')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_11_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv3_11_x1')
             .dropout(name = 'conv3_11_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_11_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv3_11_x2'))

        (self.feed('concat_3_10', 
                   'conv3_11_x2')
             .concat(3, name='concat_3_11')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_12_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv3_12_x1')
             .dropout(name = 'conv3_12_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_12_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv3_12_x2'))

        (self.feed('concat_3_11', 
                   'conv3_12_x2')
             .concat(3, name='concat_3_12')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv3_blk_bn')
             .conv(1, 1, 384, 1, 1, biased=False, relu=False,reuse=True, name='conv3_blk')
             .dropout(name = 'conv3_blk_drop', keep_prob=keep_prob_)
             .avg_pool(3, 3, 2, 2, name='pool3')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_1_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_1_x1')
             .dropout(name = 'conv4_1_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_1_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_1_x2'))

        (self.feed('pool3', 
                   'conv4_1_x2')
             .concat(3, name='concat_4_1')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_2_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_2_x1')
             .dropout(name = 'conv4_2_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_2_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_2_x2'))

        (self.feed('concat_4_1', 
                   'conv4_2_x2')
             .concat(3, name='concat_4_2')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_3_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_3_x1')
             .dropout(name = 'conv4_3_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_3_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_3_x2'))

        (self.feed('concat_4_2', 
                   'conv4_3_x2')
             .concat(3, name='concat_4_3')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_4_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_4_x1')
             .dropout(name = 'conv4_4_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_4_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_4_x2'))

        (self.feed('concat_4_3', 
                   'conv4_4_x2')
             .concat(3, name='concat_4_4')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_5_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_5_x1')
             .dropout(name = 'conv4_5_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_5_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_5_x2'))

        (self.feed('concat_4_4', 
                   'conv4_5_x2')
             .concat(3, name='concat_4_5')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_6_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_6_x1')
             .dropout(name = 'conv4_6_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_6_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_6_x2'))

        (self.feed('concat_4_5', 
                   'conv4_6_x2')
             .concat(3, name='concat_4_6')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_7_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_7_x1')
             .dropout(name = 'conv4_7_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_7_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_7_x2'))

        (self.feed('concat_4_6', 
                   'conv4_7_x2')
             .concat(3, name='concat_4_7')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_8_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_8_x1')
             .dropout(name = 'conv4_8_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_8_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_8_x2'))

        (self.feed('concat_4_7', 
                   'conv4_8_x2')
             .concat(3, name='concat_4_8')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_9_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_9_x1')
             .dropout(name = 'conv4_9_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_9_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_9_x2'))

        (self.feed('concat_4_8', 
                   'conv4_9_x2')
             .concat(3, name='concat_4_9')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_10_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_10_x1')
             .dropout(name = 'conv4_10_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_10_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_10_x2'))

        (self.feed('concat_4_9', 
                   'conv4_10_x2')
             .concat(3, name='concat_4_10')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_11_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_11_x1')
             .dropout(name = 'conv4_11_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_11_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_11_x2'))

        (self.feed('concat_4_10', 
                   'conv4_11_x2')
             .concat(3, name='concat_4_11')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_12_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_12_x1')
             .dropout(name = 'conv4_12_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_12_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_12_x2'))

        (self.feed('concat_4_11', 
                   'conv4_12_x2')
             .concat(3, name='concat_4_12')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_13_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_13_x1')
             .dropout(name = 'conv4_13_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_13_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_13_x2'))

        (self.feed('concat_4_12', 
                   'conv4_13_x2')
             .concat(3, name='concat_4_13')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_14_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_14_x1')
             .dropout(name = 'conv4_14_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_14_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_14_x2'))

        (self.feed('concat_4_13', 
                   'conv4_14_x2')
             .concat(3, name='concat_4_14')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_15_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_15_x1')
             .dropout(name = 'conv4_15_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_15_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_15_x2'))

        (self.feed('concat_4_14', 
                   'conv4_15_x2')
             .concat(3, name='concat_4_15')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_16_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_16_x1')
             .dropout(name = 'conv4_16_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_16_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_16_x2'))

        (self.feed('concat_4_15', 
                   'conv4_16_x2')
             .concat(3, name='concat_4_16')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_17_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_17_x1')
             .dropout(name = 'conv4_17_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_17_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_17_x2'))

        (self.feed('concat_4_16', 
                   'conv4_17_x2')
             .concat(3, name='concat_4_17')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_18_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_18_x1')
             .dropout(name = 'conv4_18_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_18_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_18_x2'))

        (self.feed('concat_4_17', 
                   'conv4_18_x2')
             .concat(3, name='concat_4_18')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_19_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_19_x1')
             .dropout(name = 'conv4_19_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_19_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_19_x2'))

        (self.feed('concat_4_18', 
                   'conv4_19_x2')
             .concat(3, name='concat_4_19')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_20_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_20_x1')
             .dropout(name = 'conv4_20_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_20_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_20_x2'))

        (self.feed('concat_4_19', 
                   'conv4_20_x2')
             .concat(3, name='concat_4_20')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_21_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_21_x1')
             .dropout(name = 'conv4_21_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_21_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_21_x2'))

        (self.feed('concat_4_20', 
                   'conv4_21_x2')
             .concat(3, name='concat_4_21')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_22_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_22_x1')
             .dropout(name = 'conv4_22_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_22_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_22_x2'))

        (self.feed('concat_4_21', 
                   'conv4_22_x2')
             .concat(3, name='concat_4_22')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_23_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_23_x1')
             .dropout(name = 'conv4_23_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_23_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_23_x2'))

        (self.feed('concat_4_22', 
                   'conv4_23_x2')
             .concat(3, name='concat_4_23')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_24_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_24_x1')
             .dropout(name = 'conv4_24_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_24_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_24_x2'))

        (self.feed('concat_4_23', 
                   'conv4_24_x2')
             .concat(3, name='concat_4_24')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_25_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_25_x1')
             .dropout(name = 'conv4_25_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_25_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_25_x2'))

        (self.feed('concat_4_24', 
                   'conv4_25_x2')
             .concat(3, name='concat_4_25')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_26_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_26_x1')
             .dropout(name = 'conv4_26_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_26_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_26_x2'))

        (self.feed('concat_4_25', 
                   'conv4_26_x2')
             .concat(3, name='concat_4_26')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_27_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_27_x1')
             .dropout(name = 'conv4_27_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_27_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_27_x2'))

        (self.feed('concat_4_26', 
                   'conv4_27_x2')
             .concat(3, name='concat_4_27')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_28_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_28_x1')
             .dropout(name = 'conv4_28_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_28_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_28_x2'))

        (self.feed('concat_4_27', 
                   'conv4_28_x2')
             .concat(3, name='concat_4_28')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_29_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_29_x1')
             .dropout(name = 'conv4_29_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_29_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_29_x2'))

        (self.feed('concat_4_28', 
                   'conv4_29_x2')
             .concat(3, name='concat_4_29')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_30_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_30_x1')
             .dropout(name = 'conv4_30_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_30_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_30_x2'))

        (self.feed('concat_4_29', 
                   'conv4_30_x2')
             .concat(3, name='concat_4_30')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_31_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_31_x1')
             .dropout(name = 'conv4_31_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_31_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_31_x2'))

        (self.feed('concat_4_30', 
                   'conv4_31_x2')
             .concat(3, name='concat_4_31')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_32_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_32_x1')
             .dropout(name = 'conv4_32_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_32_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_32_x2'))

        (self.feed('concat_4_31', 
                   'conv4_32_x2')
             .concat(3, name='concat_4_32')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_33_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_33_x1')
             .dropout(name = 'conv4_33_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_33_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_33_x2'))

        (self.feed('concat_4_32', 
                   'conv4_33_x2')
             .concat(3, name='concat_4_33')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_34_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_34_x1')
             .dropout(name = 'conv4_34_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_34_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_34_x2'))

        (self.feed('concat_4_33', 
                   'conv4_34_x2')
             .concat(3, name='concat_4_34')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_35_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_35_x1')
             .dropout(name = 'conv4_35_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_35_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_35_x2'))

        (self.feed('concat_4_34', 
                   'conv4_35_x2')
             .concat(3, name='concat_4_35')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_36_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv4_36_x1')
             .dropout(name = 'conv4_36_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_36_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv4_36_x2'))

        (self.feed('concat_4_35', 
                   'conv4_36_x2')
             .concat(3, name='concat_4_36')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv4_blk_bn')
             .conv(1, 1, 1056, 1, 1, biased=False, relu=False,reuse=True, name='conv4_blk')
             .dropout(name = 'conv4_blk_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_1_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_1_x1')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_1_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_1_x2'))

        (self.feed('conv4_blk', 
                   'conv5_1_x2')
             .concat(3, name='concat_5_1')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_2_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_2_x1')
             .dropout(name = 'conv5_2_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_2_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_2_x2'))

        (self.feed('concat_5_1', 
                   'conv5_2_x2')
             .concat(3, name='concat_5_2')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_3_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_3_x1')
             .dropout(name = 'conv5_3_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_3_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_3_x2'))

        (self.feed('concat_5_2', 
                   'conv5_3_x2')
             .concat(3, name='concat_5_3')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_4_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_4_x1')
             .dropout(name = 'conv5_4_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_4_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_4_x2'))

        (self.feed('concat_5_3', 
                   'conv5_4_x2')
             .concat(3, name='concat_5_4')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_5_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_5_x1')
             .dropout(name = 'conv5_5_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_5_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_5_x2'))

        (self.feed('concat_5_4', 
                   'conv5_5_x2')
             .concat(3, name='concat_5_5')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_6_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_6_x1')
             .dropout(name = 'conv5_6_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_6_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_6_x2'))

        (self.feed('concat_5_5', 
                   'conv5_6_x2')
             .concat(3, name='concat_5_6')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_7_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_7_x1')
             .dropout(name = 'conv5_7_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_7_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_7_x2'))

        (self.feed('concat_5_6', 
                   'conv5_7_x2')
             .concat(3, name='concat_5_7')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_8_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_8_x1')
             .dropout(name = 'conv5_8_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_8_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_8_x2'))

        (self.feed('concat_5_7', 
                   'conv5_8_x2')
             .concat(3, name='concat_5_8')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_9_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_9_x1')
             .dropout(name = 'conv5_9_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_9_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_9_x2'))

        (self.feed('concat_5_8', 
                   'conv5_9_x2')
             .concat(3, name='concat_5_9')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_10_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_10_x1')
             .dropout(name = 'conv5_10_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_10_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_10_x2'))

        (self.feed('concat_5_9', 
                   'conv5_10_x2')
             .concat(3, name='concat_5_10')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_11_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_11_x1')
             .dropout(name = 'conv5_11_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_11_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_11_x2'))

        (self.feed('concat_5_10', 
                   'conv5_11_x2')
             .concat(3, name='concat_5_11')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_12_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_12_x1')
             .dropout(name = 'conv5_12_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_12_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_12_x2'))

        (self.feed('concat_5_11', 
                   'conv5_12_x2')
             .concat(3, name='concat_5_12')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_13_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_13_x1')
             .dropout(name = 'conv5_13_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_13_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_13_x2'))

        (self.feed('concat_5_12', 
                   'conv5_13_x2')
             .concat(3, name='concat_5_13')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_14_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_14_x1')
             .dropout(name = 'conv5_14_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_14_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_14_x2'))

        (self.feed('concat_5_13', 
                   'conv5_14_x2')
             .concat(3, name='concat_5_14')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_15_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_15_x1')
             .dropout(name = 'conv5_15_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_15_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_15_x2'))

        (self.feed('concat_5_14', 
                   'conv5_15_x2')
             .concat(3, name='concat_5_15')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_16_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_16_x1')
             .dropout(name = 'conv5_16_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_16_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_16_x2'))

        (self.feed('concat_5_15', 
                   'conv5_16_x2')
             .concat(3, name='concat_5_16')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_17_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_17_x1')
             .dropout(name = 'conv5_17_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_17_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_17_x2'))

        (self.feed('concat_5_16', 
                   'conv5_17_x2')
             .concat(3, name='concat_5_17')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_18_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_18_x1')
             .dropout(name = 'conv5_18_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_18_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_18_x2'))

        (self.feed('concat_5_17', 
                   'conv5_18_x2')
             .concat(3, name='concat_5_18')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_19_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_19_x1')
             .dropout(name = 'conv5_19_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_19_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_19_x2'))

        (self.feed('concat_5_18', 
                   'conv5_19_x2')
             .concat(3, name='concat_5_19')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_20_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_20_x1')
             .dropout(name = 'conv5_20_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_20_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_20_x2'))

        (self.feed('concat_5_19', 
                   'conv5_20_x2')
             .concat(3, name='concat_5_20')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_21_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_21_x1')
             .dropout(name = 'conv5_21_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_21_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_21_x2'))

        (self.feed('concat_5_20', 
                   'conv5_21_x2')
             .concat(3, name='concat_5_21')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_22_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_22_x1')
             .dropout(name = 'conv5_22_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_22_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_22_x2'))

        (self.feed('concat_5_21', 
                   'conv5_22_x2')
             .concat(3, name='concat_5_22')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_23_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_23_x1')
             .dropout(name = 'conv5_23_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_23_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_23_x2'))

        (self.feed('concat_5_22', 
                   'conv5_23_x2')
             .concat(3, name='concat_5_23')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_24_x1_bn')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False,reuse=True, name='conv5_24_x1')
             .dropout(name = 'conv5_24_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_24_x2_bn')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False,reuse=True, name='conv5_24_x2'))

        (self.feed('concat_5_23', 
                   'conv5_24_x2')
             .concat(3, name='concat_5_24')
             .batch_normalization(is_training=False,reuse=True, activation_fn=tf.nn.relu, name='conv5_blk_bn')
      	     .conv(1, 1, 1024, 1, 1, biased=False, relu=False,reuse=True, name='layer1')
             .dropout(name = 'conv5_24_x1_drop', keep_prob=keep_prob_)
             .batch_normalization(is_training=False,reuse=True, activation_fn=None, name='layer1_BN')             
             .up_project([3, 3, 1024, 512], id = '2x',reuse=True, stride = 1, BN=True))
        
        up2 = self.terminals[-1]
        (self.feed('conv3_blk', 
                   up2)
             .concat(3, name='up2')
             .up_project([3, 3, 896, 584], id = '8x',reuse=True, stride = 1, BN=True))
        up3 = self.terminals[-1]
        (self.feed('conv2_blk', 
                   up3)
             .concat(3, name='up3')
             .up_project([3, 3, 776, 256], id = '16x',reuse=True, stride = 1, BN=True))
        up4 = self.terminals[-1]
        
        
        (self.feed(up4)
             .up_project([3, 3, 256, 128], id = '32x',reuse=True, stride = 1, BN=True))
        
        terminal = self.terminals[-1]
        
        print(np.shape(terminal))
        (self.feed(terminal)
             .conv(3, 3, 1, 1, 1,reuse=True, name = 'ConvPred'))
        
        depth_pred2 = self.terminals[-1]
        
        self.depth_preds.append(depth_pred2)
        
        (self.feed('data', depth_pred1)
            .concat(3, name='flow1_cat1')
            .conv(3, 3, 64, 1, 1, relu=False, name='flow1_conv1'))
        flow1_conv1_1 = self.terminals[-1]
        (self.feed(flow1_conv1_1) 
            #.batch_normalization_flow(is_training=False, activation_fn=None, name='flow1_conv1_bn')
            .conv(3, 3, 128, 2, 2, relu=False, name='flow1_conv2')
            #.batch_normalization_flow(is_training=False, activation_fn=None, name='flow1_conv2_bn')
            .conv(3, 3, 128, 1, 1, relu=False, name='flow1_conv2_2'))
        flow1_conv2_2_1 = self.terminals[-1]
        (self.feed(flow1_conv2_2_1)   
            #.batch_normalization_flow(is_training=False, activation_fn=None, name='flow1_conv2_2bn')
            .conv(3, 3, 256, 2, 2, relu=False, name='flow1_conv3')
            #.batch_normalization_flow(is_training=False, activation_fn=None, name='flow1_conv3_bn')
            .conv(3, 3, 256, 1, 1, relu=False, name='flow1_conv3_2'))
        flow1_conv3_2_1 = self.terminals[-1]
        (self.feed(flow1_conv3_2_1)
            #.batch_normalization_flow(is_training=False, activation_fn=None, name='flow1_conv3_2bn')
            .conv(3, 3, 512, 2, 2, relu=False, name='flow1_conv4')
            #.batch_normalization_flow(is_training=False, activation_fn=None, name='flow1_conv4_bn')
            .conv(3, 3, 512, 1, 1, relu=False, name='flow1_conv4_2'))
        flow1_conv4_2_1 = self.terminals[-1]
        (self.feed(flow1_conv4_2_1)
            #.batch_normalization_flow(is_training=False, activation_fn=None, name='flow1_conv4_2bn')
            .conv(3, 3, 1024, 2, 2,relu=False, name='flow1_conv5'))
            #.batch_normalization_flow(is_training=False, activation_fn=None, name='flow1_conv5_bn'))
        
        flow_tensor_1 = self.terminals[-1]
        (self.feed('data1', depth_pred2)
            .concat(3, name='flow1_cat2')
            .conv(3, 3, 64, 1, 1, relu=False,reuse=True, name='flow1_conv1'))
        flow1_conv1_2 = self.terminals[-1]
        (self.feed(flow1_conv1_2) 
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv1_bn')
            .conv(3, 3, 128, 2, 2, relu=False,reuse=True, name='flow1_conv2')
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv2_bn')
            .conv(3, 3, 128, 1, 1, relu=False,reuse=True, name='flow1_conv2_2'))
        flow1_conv2_2_2 = self.terminals[-1]
        (self.feed(flow1_conv2_2_2)        
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv2_2bn')
            .conv(3, 3, 256, 2, 2, relu=False,reuse=True,name='flow1_conv3')
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv3_bn')
            .conv(3, 3, 256, 1, 1,relu=False,reuse=True, name='flow1_conv3_2'))
        flow1_conv3_2_2 = self.terminals[-1]
        (self.feed(flow1_conv3_2_2)
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv3_2bn')
            .conv(3, 3, 512, 2, 2, relu=False,reuse=True, name='flow1_conv4')
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv4_bn')
            .conv(3, 3, 512, 1, 1,relu=False,reuse=True, name='flow1_conv4_2'))
        
        flow1_conv4_2_2 = self.terminals[-1]
        (self.feed(flow1_conv4_2_2)
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv4_2bn')
            .conv(3, 3, 1024, 2, 2,relu=False,reuse=True,name='flow1_conv5'))
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv5_bn'))
        
        flow_tensor_2 = self.terminals[-1]
        
        (self.feed('data', depth_pred1,'data1', depth_pred2)
            .concat(3, name='flow1_cat2')
            .conv(3, 3, 64, 1, 1, relu=False, name='flow1_combined'))
        flow1_conv1_combined = self.terminals[-1]
        (self.feed(flow1_conv1_combined) 
            #.batch_normalization_flow(is_training=False, activation_fn=None, name='flow1_combined_bn')
            .conv(3, 3, 128, 2, 2, relu=False,reuse=True, name='flow1_conv2')
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv2_bn')
            .conv(3, 3, 128, 1, 1, relu=False,reuse=True, name='flow1_conv2_2'))
        flow1_conv2_2_combined = self.terminals[-1]
        (self.feed(flow1_conv2_2_combined) 
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv2_2bn')
            .conv(3, 3, 256, 2, 2, relu=False,reuse=True,name='flow1_conv3')
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv3_bn')
            .conv(3, 3, 256, 1, 1,relu=False,reuse=True, name='flow1_conv3_2'))
        flow1_conv3_2_combined = self.terminals[-1]
        (self.feed(flow1_conv3_2_combined)
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv3_2bn')
            .conv(3, 3, 512, 2, 2, relu=False,reuse=True, name='flow1_conv4')
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv4_bn')
            .conv(3, 3, 512, 1, 1,relu=False,reuse=True, name='flow1_conv4_2'))
        flow1_conv4_2_combined = self.terminals[-1]
        (self.feed(flow1_conv4_2_combined)
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv4_2bn')
            .conv(3, 3, 1024, 2, 2,relu=False,reuse=True,name='flow1_conv5'))
            #.batch_normalization_flow(is_training=False, activation_fn=None,reuse=True, name='flow1_conv5_bn'))
        
        
        flow_tensor_3 = self.terminals[-1]
        
        (self.feed(flow_tensor_1,flow_tensor_2,flow_tensor_3)
            .concat(3, name='flow1_predcat')
            .conv(3, 3, 2, 1, 1, biased=False, relu=False, name='f_pred'))
        
        self.frontflows.append(self.terminals[-1])
        shape = self.terminals[-1].get_shape().as_list()
        flow_pred_up = tf.image.resize_bilinear(self.terminals[-1]*2.0, [shape[1]*2,shape[2]*2], align_corners=True)
        
        
        (self.feed(flow_tensor_1)
             .deconv(4, 4, 512, 2, 2, name = '2x_flow'))
         
        _2x_left = self.terminals[-1]
        
        (self.feed(flow_tensor_2)
             .deconv(4, 4, 512, 2, 2, name = '2x_flow',reuse=True))
         
        _2x_right = self.terminals[-1]
        
        
        (self.feed(flow_tensor_3)
            .deconv(4, 4, 512, 2, 2, name = '2x_flow',reuse=True))
        
        _2x_combined = self.terminals[-1]
        
        (self.feed(_2x_left, _2x_right, flow_pred_up)
            .smart_cat(name='smart_cat1',use_flow=True)
            .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='smart_flow1'))
        
        (self.feed('smart_flow1',_2x_combined)
        .concat(3,name='flow2_predcat')
        .conv(3, 3, 2, 1, 1, biased=False, relu=False, name='f_pred2'))
      
        self.frontflows.insert(0,self.terminals[-1]+flow_pred_up)
        shape = self.terminals[-1].get_shape().as_list()
        flow_pred2_up = tf.image.resize_bilinear(self.frontflows[0]*2.0, [shape[1]*2,shape[2]*2], align_corners=True)
        
        (self.feed(flow1_conv4_2_1,_2x_left)
             .concat(3,name='4x_left_combo')
             .deconv(4, 4, 256, 2, 2, name = '4x_flow'))
         
        _4x_left = self.terminals[-1]
        
        (self.feed(flow1_conv4_2_2,_2x_right)
             .concat(3,name='4x_right_combo')
             .deconv(4, 4, 256, 2, 2, name = '4x_flow',reuse=True))
         
        _4x_right = self.terminals[-1]
        
        
        (self.feed(flow1_conv4_2_combined,_2x_combined)
             .concat(3,name='4x_combo_combo')
             .deconv(4, 4, 256, 2, 2, name = '4x_flow',reuse=True))
        
        _4x_combined = self.terminals[-1]
        
        
        
        (self.feed(_4x_left, _4x_right, flow_pred2_up)
            .smart_cat(name='smart_cat2',use_flow=True)
            .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='smart_flow2')) 
        
        (self.feed('smart_flow2',_4x_combined)
        .concat(3,name='flow3_predcat')
        .conv(3, 3, 2, 1, 1, biased=False, relu=False, name='f_pred3'))  
        
        self.frontflows.insert(0,self.terminals[-1]+flow_pred2_up)        
       
        shape = self.terminals[-1].get_shape().as_list()
        flow_pred3_up = tf.image.resize_bilinear(self.frontflows[0]*2.0, [shape[1]*2,shape[2]*2], align_corners=True)
        
        
        (self.feed(flow1_conv3_2_1,_4x_left)
             .concat(3,name='8x_left_combo')
             .deconv(4, 4, 128, 2, 2, name = '8x_flow'))
         
        _8x_left = self.terminals[-1]
        
        (self.feed(flow1_conv3_2_2,_4x_right)
             .concat(3,name='8x_left_combo')
             .deconv(4, 4, 128, 2, 2, name = '8x_flow',reuse=True))
         
        _8x_right = self.terminals[-1]
        
        
        (self.feed(flow1_conv3_2_combined,_4x_combined)
             .concat(3,name='8x_combo_combo')
             .deconv(4, 4, 128, 2, 2, name = '8x_flow',reuse=True))
        
        _8x_combined = self.terminals[-1]
        
        
        (self.feed(_8x_left, _8x_right, flow_pred3_up)
            .smart_cat(name='smart_cat3',use_flow=True)
            .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='smart_flow3'))
        
        (self.feed('smart_flow3',_8x_combined)
        .concat(3,name='flow4_predcat')
        .conv(3, 3, 2, 1, 1, biased=False, relu=False, name='f_pred4'))
        
        
        self.frontflows.insert(0,self.terminals[-1]+flow_pred3_up)        
       
        shape = self.terminals[-1].get_shape().as_list()
        flow_pred4_up = tf.image.resize_bilinear(self.frontflows[0]*2.0, [shape[1]*2,shape[2]*2], align_corners=True)
       
        (self.feed(flow1_conv2_2_1,_8x_left)
             .concat(3,name='16x_left_combo')
             .deconv(4, 4, 64, 2, 2, name = '16x_flow'))
         
        _16x_left = self.terminals[-1]
        
        (self.feed(flow1_conv2_2_2,_8x_right)
             .concat(3,name='16x_right_combo')
             .deconv(4, 4, 64, 2, 2, name = '16x_flow',reuse=True))
         
        _16x_right = self.terminals[-1]
        
        
        (self.feed(flow1_conv2_2_combined,_8x_combined)
             .concat(3,name='16x_combo_combo')
             .deconv(4, 4, 64, 2, 2, name = '16x_flow',reuse=True))
        
        _16x_combined = self.terminals[-1]

      
        (self.feed(_16x_left, _16x_right, flow_pred4_up)
            .smart_cat(name='smart_cat4',use_flow=True)
            .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='smart_flow4'))
        
        (self.feed('smart_flow4',_16x_combined)
        .concat(3,name='flow5_predcat')
        .conv(3, 3, 2, 1, 1, biased=False, relu=False, name='f_pred5'))
           
        
        self.frontflows.insert(0,self.terminals[-1]+flow_pred4_up)
        
	       
        
        (self.feed('flow1_predcat')
        .conv(3, 3, 3, 1, 1, biased=False, relu=False, name='sigma_pred1'))
        
        sigma_pred1_up = tf.image.resize_bilinear(self.terminals[-1], [192,640], align_corners=True)
        
        (self.feed('flow2_predcat')
        .conv(3, 3, 3, 1, 1, biased=False, relu=False, name='sigma_pred2'))
        
        sigma_pred2_up = tf.image.resize_bilinear(self.terminals[-1], [192,640], align_corners=True)
        
        
        (self.feed('flow3_predcat')
        .conv(3, 3, 3, 1, 1, biased=False, relu=False, name='sigma_pred3'))
        
        sigma_pred3_up = tf.image.resize_bilinear(self.terminals[-1], [192,640], align_corners=True)
        
        (self.feed('flow4_predcat')
        .conv(3, 3, 3, 1, 1, biased=False, relu=False, name='sigma_pred4'))
        
        sigma_pred4_up = tf.image.resize_bilinear(self.terminals[-1], [192,640], align_corners=True)
        
        (self.feed('flow5_predcat')
        .conv(3, 3, 3, 1, 1, biased=False, relu=False, name='sigma_pred5'))
        
        sigma_pred5_up = tf.image.resize_bilinear(self.terminals[-1], [192,640], align_corners=True)
                
        (self.feed(sigma_pred1_up,sigma_pred2_up,sigma_pred3_up,sigma_pred4_up,sigma_pred5_up)
        .concat(3,name='sigma_predcat')
        .conv(3, 3, 3, 1, 1, biased=False, relu=False, name='sigma_final'))
        
        
