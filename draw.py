import math

import cv2
import numpy as np


previous_position = []
#Front View
theta, phi = 3.1415/24, -3.1415/50
#Diagonal View
# theta, phi = 3.1415/12, -3.1415/40
#Top View
# theta, phi = 3.1415/12, -3.1415/4
should_rotate = False
scale_dx = 800
scale_dy = 800


class Plotter3d:
    SKELETON_EDGES = np.array([[0,1], [0,2], [1,3], [2,4], [5,7], [7,9], [6,8], [8,10], [5,6], 
                               [5,11], [11,13], [13,15], [6,12], [12,14], [14,16], [11,12]])

    # SKELETON_EDGES = np.array([[11, 10], [10, 9], [9, 0], [0, 3], [3, 4], [4, 5], [0, 6], [6, 7], [7, 8], [0, 12],
    #                            [12, 13], [13, 14], [0, 1], [1, 15], [15, 16], [1, 17], [17, 18]])

    def __init__(self, canvas_size, origin=(0.8, 0.5), scale=2):
        self.origin = np.array([origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32)  # x, y
        self.scale = np.float32(scale)
        self.theta = 0
        self.phi = 0
        axis_length = 200
        axes = [
            np.array([[-axis_length/2, -axis_length/2, 0], [axis_length/2, -axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, -axis_length/2, axis_length]], dtype=np.float32)]
        step = 20
        for step_id in range(axis_length // step + 1):  # add grid
            axes.append(np.array([[-axis_length / 2, -axis_length / 2 + step_id * step, 0],
                                  [axis_length / 2, -axis_length / 2 + step_id * step, 0]], dtype=np.float32))
            axes.append(np.array([[-axis_length / 2 + step_id * step, -axis_length / 2, 0],
                                  [-axis_length / 2 + step_id * step, axis_length / 2, 0]], dtype=np.float32))
        self.axes = np.array(axes)

    def plot(self, img, vertices, edges):
        global theta, phi
        img.fill(0)
        R = self._get_rotation(theta, phi)
        self._draw_axes(img, R)
        if len(edges) != 0:
            self._plot_edges(img, vertices, edges, R)

    def _draw_axes(self, img, R):
        axes_2d = np.dot(self.axes, R)
        axes_2d = axes_2d * self.scale + self.origin
        for axe in axes_2d:
            axe = axe.astype(int)
            cv2.line(img, tuple(axe[0]), tuple(axe[1]), (128, 128, 128), 1, cv2.LINE_AA)

    def _plot_edges(self, img, vertices, edges, R):
        vertices_2d = np.dot(vertices, R)
        edges_vertices = vertices_2d.reshape((-1, 2))[edges] * self.scale + self.origin

        colors = [(255,64,64), (138,43,226), (152,245,255), (255,97,3), (255,127,80), (191,62,255),
        (255,64,64), (138,43,226), (152,245,255), (255,97,3), (255,127,80), (191,62,255),
        (255,64,64), (138,43,226), (152,245,255), (255,97,3), (255,127,80), (191,62,255)]
        
        for idx, edge_vertices in enumerate(edges_vertices):
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(edge_vertices[1]), (colors[idx][-1],colors[idx][1],colors[idx][0]), 2, cv2.LINE_AA)

    def _get_rotation(self, theta, phi):
        sin, cos = math.sin, math.cos
        return np.array([
            [ cos(theta),  sin(theta) * sin(phi)],
            [-sin(theta),  cos(theta) * sin(phi)],
            [ 0,                       -cos(phi)]
        ], dtype=np.float32)  # transposed

    @staticmethod
    def mouse_callback(event, x, y, flags, params):
        global previous_position, theta, phi, should_rotate, scale_dx, scale_dy
        if event == cv2.EVENT_LBUTTONDOWN:
            previous_position = [x, y]
            should_rotate = True
        if event == cv2.EVENT_MOUSEMOVE and should_rotate:
            theta += (x - previous_position[0]) / scale_dx * 6.2831  # 360 deg
            phi -= (y - previous_position[1]) / scale_dy * 6.2831 * 2  # 360 deg
            phi = max(min(3.1415 / 2, phi), -3.1415 / 2)
            previous_position = [x, y]
        if event == cv2.EVENT_LBUTTONUP:
            should_rotate = False

# AIST_COCO_MAP = {
#     'Nose': 0,
#     'LEye': 1,
#     'REye': 2,
#     'LEar': 3,
#     'REar': 4,
#     'LShoulder': 5,
#     'RShoulder': 6,
#     'LElbow': 7,
#     'RElbow': 8,
#     'LWrist': 9,
#     'RWrist': 10,
#     'LHip': 11,
#     'RHip': 12,
#     'LKnee': 13,
#     'RKnee': 14,
#     'LAnkle': 15,
#     'RAnkle': 16
# }

body_edges = np.array(
    [ [0,1], [0,2], # Nose - LEye, Nose - REye
     [1,3], [2,4], # LEye - LEar, REye - REar
     [5,7], [7,9], # LShoulder - LElbow, LElbow - LWrist
     [6,8], [8,10], # RShoulder - RElbow, RElbow - RWrist
     [5,6], # LShoulder - RShoulder
     [5,11], [11,13], [13,15], # LShoulder - LHip, LHip - LKnee, LKnee - LAnkle
     [6,12], [12,14], [14,16], # RShoulder - RHip, RHip - RKnee, RKnee - RAnkle
     [11,12] # LHip - RHip
    ]
)

def draw_poses(img, poses_2d):
    for pose_id in range(len(poses_2d)):
        pose = np.array(poses_2d[pose_id][0:-1]).reshape((-1, 3)).transpose()
        was_found = pose[2, :] > 0
        for edge in body_edges:
            if was_found[edge[0]] and was_found[edge[1]]:
                cv2.line(img, tuple(pose[0:2, edge[0]].astype(int)), tuple(pose[0:2, edge[1]].astype(int)),
                         (255, 255, 0), 4, cv2.LINE_AA)
        for kpt_id in range(pose.shape[1]):
            if pose[2, kpt_id] != -1:
                cv2.circle(img, tuple(pose[0:2, kpt_id].astype(int)), 3, (0, 255, 255), -1, cv2.LINE_AA)
