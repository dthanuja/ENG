#!/bin/bash

wget https://www.dropbox.com/s/0pus4p0mpaeii5f/model.ckpt-60000.data-00000-of-00001 -P  depth_flow_pose_models/snapshots
wget https://www.dropbox.com/s/jr5fct6qzgg1lbg/model.ckpt-60000.index -P depth_flow_pose_models/snapshots
wget https://www.dropbox.com/s/d98rjgcmzhh04s1/model.ckpt-60000.meta -P depth_flow_pose_models/snapshots
mkdir pose_preds
