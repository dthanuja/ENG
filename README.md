# ENG
ENG: End-to-end neural geometry for robust depth and pose estimation using CNNs

<p align="center">
  <img src="eng_demo.gif" alt="eng">
</p>

### Requirements

The code is tested on CUDA 9.0 + cudNN 7.4.2 + Python 3.5.2 + Tensorflow 1.9

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

### Single Image Depth Prediction Results (KITTI 0-50m)

| RMSE (m)    | Rel<sub>abs</sub>|Accuracy (&#948;)  | Accuracy (&#948;<sup>2</sup>) | Accuracy (&#948;<sup>3</sup>)|
| ----------- |------------------| ---------------------|---------------------------------|---------------------------------|
|  <p align="center"> 3.284  </p>    |    <p align="center">0.092  </p>       | 	<p align="center">90.6% 	</p>	| 		<p align="center">97.1%	</p>	  | 		<p align="center">98.9%	</p>	    |


### Single Image Depth Prediction Results (NYUv2 [using the indoor model] )

| RMSE (m)    | Rel<sub>abs</sub>| Accuracy (&#948;)  | Accuracy (&#948;<sup>2</sup>) | Accuracy (&#948;<sup>3</sup>)|
| ----------- |------------------| ---------------------|---------------------------------|---------------------------------|
|   <p align="center">0.478    </p>   |   <p align="center"> 0.111  </p>         | <p align="center">	87.2% 		| 		<p align="center">97.8%		 </p>   | 		<p align="center">99.5%  </p> 	    | 


### Pose Estimation  (KITTI)

| Sequence    | ATE (m) | RPE (m) |   RPE (&#176;) |  
| ----------- |------------------| ---------------------| ---------------------|
|   <p align="center">9    </p>   |   <p align="center"> 16.55</p>         | <p align="center">	0.047	</p> |<p align="center">	0.128	</p> 
|  <p align="center">10  </p> | <p align="center"> 9.846</p>         | <p align="center">	0.039	</p> | <p align="center">	0.138	</p>|
