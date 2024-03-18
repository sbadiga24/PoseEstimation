import cv2
from OdModel import *
import keyboard
camera_sn=25062645
# camera = Camera(camera_id=1,camera_sn=camera_sn)
model_path="PoseEstimation\model\yolov8l_tennis_ball.pt"
camera1 = ObjectDetection(model_path,camera_id=1,camera_sn=camera_sn)

print(f"Stereo Camera Sn-{camera_sn}, Parameters: BaseLine-{camera1.camera.Baseline} F_pixel-{camera1.camera.f_pixel} CxCy-{camera1.camera.CxCy}")
camera1.process_images()
    # cv2.waitKey(1)
