import pickle
import numpy as np

with open('./gBR_sBM_cAll_d04_mBR0_ch03.pkl','rb') as f:
    data = pickle.load(f)

coco_kpts = data['keypoints3d_optim']

print(coco_kpts.shape)

with open('default_aist.npy','wb') as f:
    np.save(f,coco_kpts)