git clone https://github.com/google/aistplusplus_api.git
cd aistplusplus_api
pip install -r requirements.txt
python setup.py install
cd aistplusplus_api
mkdir keypoints_dir
mkdir smpl_body
cd keypoints_dir
mkdir keypoints3d
mkdir motions
mkdir cameras
cd ../../
mv mapping.txt ./aistplusplus_api/keypoints_dir/cameras/
mv setting1.json ./aistplusplus_api/keypoints_dir/cameras/
pip install chumpy
mv ./aistplusplus_api ./COCO_to_SMPL_Estimation