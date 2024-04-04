import cv2
import numpy as np
from CamParam import *

class Camera:
    def __init__(self,camera_id,camera_sn,resolution='HD',desired_fps=60,cam=True):
        self.camera_id= camera_id
        self.cam=cam
        self.resolution = {"2K": ["2k", 4416, 1242], "FHD": ["FHD", 3840, 1080], "HD": ["HD", 2560, 720], "VGA": ["VGA", 1344, 376]}
        self.SetResolution= self.resolution[resolution]
        self.desired_fps=desired_fps
        self.cap= self.init_camera()


        StCamera=CamParam(camera_sn=camera_sn)
        StCamera.get_params()
        
        self.Baseline=StCamera.Baseline
        self.f_pixel=StCamera.f_pixel
        self.CxCy=StCamera.CxCy
        

    def init_camera(self):
        if self.cam:
            cap = cv2.VideoCapture(self.camera_id)#r'Utils\vd11_1.mp4'
        else:
            cap = cv2.VideoCapture(r'data\videos\vd1.mp4')
            
        if not cap.isOpened():
            print(f"Error opening video stream for camera {self.camera_id}")
            exit(-1)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.SetResolution[1])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.SetResolution[2])
            # Attempt to set the FPS
            cap.set(cv2.CAP_PROP_FPS, self.desired_fps)

            # Optionally, check if the FPS was set correctly
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Requested FPS: {self.desired_fps}, Actual FPS: {actual_fps}")
            return cap
        

    def capture_frame(self):
         retval, frame = self.cap.read()
         if retval:
            left_image, right_image = np.split(frame, 2, axis=1)
            return left_image, right_image
         else:
            print("Failed to capture frame.")
            return None, None
    def release(self):
        self.cap.release()