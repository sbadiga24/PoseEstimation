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
    zDepth = (baseline*f_pixel)/disparity             #Depth in [cm]
    X=((x2-CxCy['Left_Cx'])*zDepth)/f_pixel
    Y=((y2-CxCy['Left_Cy'])*zDepth)/f_pixel

    return [X,Y,zDepth]


