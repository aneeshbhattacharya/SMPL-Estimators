import pickle
import numpy as np

with open('./aist_sample.pkl','rb') as f:
    data = pickle.load(f)

smpl_poses = data['smpl_poses']
smpl_trans = data['smpl_trans']

smpl_poses_list = []
for i in range(0,smpl_poses.shape[0],2):
    smpl_poses_list.append(smpl_poses[i])

smpl_trans_list = []
for i in range(0, smpl_trans.shape[0],2):
    smpl_trans_list.append(smpl_trans[i])

data['smpl_poses'] = np.array(smpl_poses_list)
data['smpl_trans'] = np.array(smpl_trans_list)

with open('./aist_downsampled.pkl','wb') as f:
    pickle.dump(data,f)
