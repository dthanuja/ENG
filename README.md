# ENG
ENG: End-to-end neural geometry for robust depth and pose estimation using CNNs

<p align="center">
  <img src="eng_demo.gif" alt="eng">
</p>

### Requirements

The code is tested on CUDA 9.0 + Python 3.5.2 + Tensorflow 1.9

Download the weights (requires wget)
``` bash
./download_model.sh
```

### Running 

form the root dir of the project run:

``` bash
python -m test_depth_flow_pose
```

#### Run Options

There are some configuration parameters, use the following command to print the help menu

``` bash
python -m test_depth_flow_pose --help
```


### Citing the Paper(s)

If you found this repository useful please cite the following:

```
@article{dharmasiri2018eng,
  title={ENG: End-to-end Neural Geometry for Robust Depth and Pose Estimation using CNNs},
  author={Dharmasiri, Thanuja and Spek, Andrew and Drummond, Tom},
  journal={arXiv preprint arXiv:1807.05705},
  year={2018}
}

```

