# Getting started
This code has been assembled from the following 3 repositories: <br>
<ol>
<li>COCO to SMPL Estimator: https://github.com/google/aistplusplus_api</li>
<li>AMASS to SMPL Estimator: https://github.com/GuyTevet/motion-diffusion-model</li>
<li>Visualize 3D Keypoints: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch</li>
</ol>

The code has been tested on Ubuntu 20.04 and requires: <br>
<ol>
<li>Python 3.7</li>
<li>Conda</li>
<li>CUDA capable GPU</li>
</ol>

### Setup for keypoints visualizer
Set up the Anaconda environment: <br>
```
conda create -n VisDemo python=3.7
conda activate VisDemo
./setup_visualizer.sh
```

### Visualize 3D Keypoints

```
cd lightweight-human-pose-estimation-3d-demo.pytorch 
python plot_AIST_Keypoints.py --joints ../aist_sample.npy
```

## COCO 3D Body Keypoints to SMPL Estimation
