import sys
import cv2
import numpy as np
import time


def pose(right_point, left_point, baseline,f_pixel,CxCy,metric="cm"):
    #convert to meters or cm or ft
    if metric == 'm':
        baseline =  baseline/1000
        f=2.1/1000     
        
    elif metric == 'cm':
        baseline =  baseline/10   
        f=2.1/10
    elif metric == 'ft':
        baseline=   baseline / 304.8     
        f=2.1/304.8
    
    x1,y1 = right_point
    x2,y2 = left_point

    # CALCULATE THE DISPARITY:
    disparity = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
     #Displacement between left and right frames [pixels]
    if disparity == 0:
        print("Warning: Disparity is zero, which may indicate an error in point matching.")
        return None, None, None 
    # CALCULATE DEPTH z:
    zDepth = round(((baseline*f_pixel)/disparity),2)             #Depth in [cm]
    X=round((((x2-CxCy['Left_Cx'])*zDepth)/f_pixel),2)
    Y=round((((y2-CxCy['Left_Cy'])*zDepth)/f_pixel),2)
    
    point_camera_frame=[X,Y,zDepth]
    camera_position = np.array([30,0,30])
    point_camera_frame = np.array([X,Y,zDepth])
    # Construct the transformation matrix
    T = construct_transformation_matrix(camera_position)

    # Transform point to world frame
    point_world_frame = transform_point_homogeneous(point_camera_frame, T)#
    return point_world_frame #[X,Y,zDepth]#

def construct_transformation_matrix(camera_position):
    # Rotation matrix: 180 degrees around Z (reverses X), then swap Y and Z
    R = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]])
    t = camera_position
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1
    return T

def transform_point_homogeneous(point_camera_frame, T):
    # Convert point to homogeneous coordinates
    point_homogeneous = np.append(point_camera_frame, 1)
    # Apply transformation
    point_world_frame_homogeneous = np.dot(T, point_homogeneous)
    # Convert back to Cartesian coordinates
    point_world_frame = point_world_frame_homogeneous[:3] / point_world_frame_homogeneous[3]
    return point_world_frame

