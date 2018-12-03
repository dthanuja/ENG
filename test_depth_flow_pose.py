"""
Created on Tue Apr 11 18:22:35 2017

@author: thanuja
# """
import argparse
import os

import numpy as np
import tensorflow as tf
from TestDataset import DataSetNew as TestDataSet

from utils import inv_preprocess,prepare_depth,prepare_flow,prepare_info

import scipy.io as sio
import depth_flow_pose as VO
import vo_utils as vo_utils
import matplotlib.pyplot as plt

BATCH_SIZE = 1

SNAPSHOT_DIR = 'depth_flow_pose_models/snapshots'
SNAPSHOT = 'model.ckpt-60000'

OUTPUT_DIR = 'pose_preds'
TEST_CSV = 'test.csv'
PLOT_TESTS=True
SAVE_TESTS=True

# parse command line arguments
def get_arguments():
    parser = argparse.ArgumentParser(description="ENG:end-to-end neural geometry network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")

    # MODEL
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--snapshot", type=str, default=SNAPSHOT,
                        help="Name of the snapshot file.")
    
    # TESTING
    parser.add_argument("--test-output-dir", type=str, default=OUTPUT_DIR,
                        help="Where to save the test outputs.")
    parser.add_argument("--test-csv", type=str, default=TEST_CSV,
                        help="Test .csv to infer on.")
    parser.add_argument("--plot-test", type=bool, default=PLOT_TESTS,
                        help="Plots a visualisation of the resulting predictions (press Q to exit viewer)")
    parser.add_argument("--save-test-results", type=bool, default=SAVE_TESTS,
                        help="Plots a visualisation of the resulting predictions (press Q to exit viewer)")
    return parser.parse_args()


# load model params
def load(saver, sess):
    saver.restore(sess, SNAPSHOT_DIR+'/model.ckpt-60000')
    print("Restored model parameters")


# plots the results for visualisation
def plotResults(left_rgb, right_rgb, depth, flow):
    fig = plt.figure()
    #fig.tight_layout()

    # Input images are mean subtracted - mean is required to colour correct for visualisation
    mean_arr = np.array([91.090, 95.435,  96.119])

    # RGB LEFT
    a=fig.add_subplot(2,2,1)
    imgplot = plt.imshow(np.squeeze((left_rgb+mean_arr)/255))
    a.set_title('RGB (Time : t0)')
    
    # RGB RIGHT
    a=fig.add_subplot(2,2,2)
    imgplot = plt.imshow(np.squeeze((right_rgb+mean_arr)/255))
    a.set_title('RGB (Time : t1)')
    
    # DEPTH IMAGE
    a=fig.add_subplot(2,2,3)
    depth = prepare_depth(depth, 1)
    imgplot = plt.imshow(np.squeeze(depth))
    a.set_title('Predicted Depth')

    # FLOW IMAGE
    a=fig.add_subplot(2,2,4)
    
    # flow is 2-channel convert to a red-green colour flow image
    flow_arr = prepare_flow(flow,1,100)
    imgplot = plt.imshow(np.squeeze(flow_arr))
    a.set_title('Preicted Flow')
    
    # Reduce the default white space 
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.0, wspace=0.05)
    
    # Generates plot
    plt.show()


# saves the resulting network outputs to file
def saveResultsToFile( depth, flow, alpha, beta, gamma, predicted_pose, idx=0, outdir=OUTPUT_DIR, step=0):

    # save pose to text file
    poses_file = open('%s/poses_%d.txt'%(outdir, idx),'a')
    np.savetxt(poses_file,[predicted_pose],fmt="%f")
    poses_file.close()

    # save the depth and flow at .mats, readable in python, matlab, etc...
    sio.savemat('%s/depth_%06d.mat' %(outdir, step),{"depth":depth})
    sio.savemat('%s/flow_%06d.mat' %(outdir, step),{"flow":flow})

    # the confidence matrix is symmetric and is saved seperately for each element of the upper right triangle
    # please see the paper for more details of the symbol meanings
    sio.savemat('%s/confidence_alpha_%06d.mat' %(outdir, step),{"a":alpha})
    sio.savemat('%s/confidence_beta_%06d.mat' %(outdir, step),{"b":beta})
    sio.savemat('%s/confidence_gamma_%06d.mat' %(outdir, step),{"c":gamma})


# creates network and splits images into inputs
def createNetwork(image_stack, batch_size=1, training=False):
    # split stack back into individual images for inference
    rgb_left = image_stack[:,:,:,0:3]
    rgb_left_mod = rgb_left/50.0
    rgb_right = image_stack[:,:,:,3:6]
    rgb_right_mod = rgb_right/50.0

    net =  VO.depthflowPoseNet({'data_':rgb_left,'data1_':rgb_right,'data': rgb_left_mod,'data1': rgb_right_mod},batch=batch_size,is_training=training)

    return net, rgb_left, rgb_right


# runs inference on the network
def getNetworkOutputs(sess, net, cameras, img_sizes, batch_idx=0):
    # ORIGINAL INPUT IMAGE SIZE
    height = img_sizes[batch_idx, 0]
    width = img_sizes[batch_idx, 1]

    camera = tf.expand_dims( cameras[batch_idx,:,:],0)

    # PREDICTED DEPTH
    pred_depth_batch = net.depth_preds[0]    
    pred_depth = pred_depth_batch[batch_idx,:,:,:]
    pred_depth = tf.expand_dims(pred_depth,0)
    pred_depth = tf.image.resize_bilinear(pred_depth,(height,width),align_corners=True)

    # PREDICTED FLOW
    pred_flow_batch = net.frontflows[0]
    pred_flow = pred_flow_batch[batch_idx,:,:,:]
    
    # fixed at double
    flow_scale = 2.0 

    # rescale due to upsample
    pred_flow =  tf.expand_dims(pred_flow*flow_scale,0)
    pred_flow = tf.image.resize_bilinear(pred_flow,(height,width),align_corners=True)
    
    pred_a_batch, pred_b_batch, pred_c_batch = net.get_uncertainty(img_sizes)        
    
    c_x = pred_a_batch[batch_idx]
    c_xy = pred_b_batch[batch_idx]
    c_y = pred_c_batch[batch_idx]

    # predict pose using predicted values
    predicted_pose =vo_utils.compute_pose(
        depths=pred_depth,
        flow=pred_flow,
        c_x=c_x,
        c_y=c_y,
        c_xy=c_xy,
        batchsize=1,
        k=camera) 
    predicted_pose = predicted_pose[batch_idx,:,0]

    # perform inference 
    (depth_out, flow_out, alpha, beta, gamma, pose_out) = sess.run([pred_depth, pred_flow, c_x, c_xy, c_y, predicted_pose])

    return (depth_out, flow_out, alpha, beta, gamma, pose_out)


def infer_depth_flow_pose(args,startidx,endidx,idx):
    
    # setup dataset readed to load images from csvfile
    dataset2 = TestDataSet(1,'Train',gen_log=False)
    images,cams_batch,shapes_batch = dataset2.csv_inputs(args.test_csv)

    # create empty file if already exists
    poses_file = open('%s/poses_%d.txt'%(args.test_output_dir,idx),'w')
    poses_file.close()
    
    net, rgb_left_tensor, rgb_right_tensor = createNetwork(images)

    with tf.Session() as sess:
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        restore_var = [v for v in tf.global_variables()]
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess)
        
       
        device_name = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'

        # device_name = '/cpu:0'
        with tf.device(device_name):
             
            for step in range(startidx,endidx):

                print('Running inference : {}'.format(step))

                (rgb_left, rgb_right) = sess.run([rgb_left_tensor, rgb_right_tensor])
                (depth, flow, alpha, beta, gamma, pose_out) = getNetworkOutputs(sess, net, cams_batch, shapes_batch)

                with tf.device('/cpu:%d' % 0):
                    if args.plot_test:
                        plotResults(rgb_left, rgb_right, depth, flow)
                    if args.save_test_results:
                        saveResultsToFile(depth, flow, alpha, beta, gamma, pose_out, idx, args.test_output_dir, step)

        print('Inference Complete!')
                    
        coord.request_stop()
        coord.join(threads)

         
if __name__ == '__main__':
#    main()
   args = get_arguments()

   print('Reading csvfile: %s'%args.test_csv);
   
   infer_depth_flow_pose(args,0,1,0)
#                        
             
        


