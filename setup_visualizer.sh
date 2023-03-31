git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch.git
sleep 3
cd lightweight-human-pose-estimation-3d-demo.pytorch
pip install -r requirements.txt
pip install moviepy
rm -rf ./modules/draw.py
mv ../draw.py ./modules/
mv ../plot_AIST_Keypoints.py ./