# Getting started
This code has been assembled from the following 3 repositories: <br>
<ol>
<li>COCO to SMPL Estimator: https://github.com/google/aistplusplus_api</li>
<li>Visualize 3D Keypoints: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch</li>
<li>SMPL-to-FBX: https://github.com/softcat477/SMPL-to-FBX.git</li>
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
setup_visualizer.sh
```

### Visualize 3D Keypoints

```
cd lightweight-human-pose-estimation-3d-demo.pytorch 
python plot_AIST_Keypoints.py --joints ../aist_sample.npy
```

## COCO 3D Body Keypoints to SMPL Estimation

```
conda create -n COCO_Estimator python=3.7
conda activate COCO_Estimator
setup_coco.sh
```
Further steps:
<ol>
<li>Download SMPL body model: https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip and place the "m" body as SMPL_MALE.pkl in COCO_to_SMPL_Estimation/smpl_body/</li>
<li>Replace aist_plusplus_api-1.1.0-py3.7.egg in the created Anaconda env in site-packages/ with the one provided in this repository</li>
<li>Place the aist_sample.pkl in the COCO_to_SMPL_Estimation/keypoints_dir/keypoints3d</li>
</ol>

Command to estimate: 
```
python processing/run_estimate_smpl.py --anno_dir ./keypoints_dir/ --smpl_dir ./smpl_body/ --save_dir ./keypoints_dir/motions/ --data_type internal
```

## SMPL Motion to FBX
```
conda create -n FBX python=3.7
conda activate FBX
setup_fbx.sh
```
Setting up Python FBX: 
<ol>
<li>Download Python FBX SDK: https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3</li>
<li>Extract from the folder</li>
<li>Run and extract the FBX files into your directory (Eg. MY_FBX_FILES)</li>
<li>Copy the files to site-packages of the conda env</li>
</ol>

Further steps:
<ol>
<li>Download the SMPL fbx model for Unity from: https://smpl.is.tue.mpg.de/</li>
<li>Place inside SMPL-to-FBX</li>
</ol>

Run code:
```
python3 Convert.py --input_pkl_base Pkls/aist_downsampled.pkl --fbx_source_path ./SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx --output_base ./

```