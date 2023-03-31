from argparse import ArgumentParser
import json
import os

import cv2
import numpy as np
import time

# from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses
import moviepy.editor as mpe
import statistics

from argparse import ArgumentParser


np.set_printoptions(precision=2)

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d

parser = ArgumentParser(description=''
                                    'Press esc to exit, "p" to (un)pause video or process next image.')

parser.add_argument('--joints', '-j', help='Mandatory path to joints numpy', type=str, required=True)
parser.add_argument('--visualize', '-v', help='Visualize the keypoints in real time', type=str, default="True")
parser.add_argument('--name', '-n',
                    help='Save_Name',
                    type=str,default='./rendered.mp4')

args = parser.parse_args()

joints_path = args.joints
mp4_save_name = args.name

visualize_boolean = args.visualize

if visualize_boolean.lower() == "false" or visualize_boolean.lower() == "f":
    visualize_boolean = False
else:
    visualize_boolean = True 

canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
plotter = Plotter3d(canvas_3d.shape[:2])
canvas_3d_window_name = 'Canvas 3D'
cv2.namedWindow(canvas_3d_window_name)
cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

file_path = None
if file_path is None:
    file_path = os.path.join('data', 'extrinsics.json')
with open(file_path, 'r') as f:
    extrinsics = json.load(f)
R = np.array(extrinsics['R'], dtype=np.float32)
t = np.array(extrinsics['t'], dtype=np.float32)

base_height = 256


delay = 1
esc_code = 27
p_code = 112
space_code = 32
mean_time = 0
is_video = False

with open(joints_path,'rb') as f:
    pose_data = np.load(f)

start = time.time()

# Interpolate and smoothen poses
joint_map = {}
for pose3d in pose_data:

    pose3d = pose3d.reshape(17,3)

    for i in range(pose3d.shape[0]):
        if i not in joint_map.keys():
            joint_map[i] = {
                'x':[],
                'y':[],
                'z':[]
            }

        joint_map[i]['x'].append(pose3d[i][0])        
        joint_map[i]['y'].append(pose3d[i][1])
        joint_map[i]['z'].append(pose3d[i][2])


reconstruct = np.zeros((pose_data.shape[0],17,3))

for i in range(reconstruct.shape[0]):
    temp = np.zeros((17,3))
    for joints in joint_map.keys():

        temp[joints] = [joint_map[joints]['x'][i],joint_map[joints]['z'][i],joint_map[joints]['y'][i]] 

    reconstruct[i] = temp

pose_data = reconstruct
counter = 0

print(os.path.exists('./temp_images'))

if not os.path.exists('/temp_images'):
    os.makedirs('./temp_images')

if visualize_boolean:

    for idx, poses_3d in enumerate(pose_data):

        poses_3d = poses_3d.reshape(1,17,3)
        edges = []

        if len(poses_3d):
            poses_3d = poses_3d.reshape(poses_3d.shape[0], 17, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 17 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        plotter.plot(canvas_3d, poses_3d, edges)
        cv2.imshow(canvas_3d_window_name, canvas_3d)

        frame = canvas_3d

        cv2.imwrite('./temp_images/'+str(counter)+".jpg",canvas_3d)
        counter+=1

        key = cv2.waitKey(100)

else:

    for idx, poses_3d in enumerate(pose_data):

        poses_3d = poses_3d.reshape(1,17,3)
        edges = []

        if len(poses_3d):
            poses_3d = poses_3d.reshape(poses_3d.shape[0], 17, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 17 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        plotter.plot(canvas_3d, poses_3d, edges)
        frame = canvas_3d

        cv2.imwrite('./temp_images/'+str(counter)+".jpg",canvas_3d)
        counter+=1

end_time = time.time()

print("Time taken: {}".format(end_time-start))

height,width = frame.shape[0], frame.shape[1]

video = cv2.VideoWriter(mp4_save_name, 0x7634706d, 60, (width, height))

for i in range(pose_data.shape[0]):
    file_name = './temp_images/'+str(i)+'.jpg'
    frame = cv2.imread(file_name)
    video.write(frame)
    os.system("rm -rf {}".format(file_name))

os.system("rm -rf {}".format('./temp_images'))


