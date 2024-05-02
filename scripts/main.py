import os
import sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Add the 'utils' folder to sys.path
sys.path.append(utils_path)

import threading
from time import sleep

import cv2
# from OdModel import *
from contour_OBdetect import *
import keyboard
camera1_sn=25062645 #right
camera2_sn=20778657 #left




if __name__ == "__main__":
    camera1 = contour_OB(camera_id=1,camera_sn=camera2_sn,cam_state=False,N_prediction=150)

    
    # camera2 = contour_OB(camera_id=2,camera_sn=camera2_sn)

    # camera2 = ObjectDetection(model_path,camera_id=1,camera_sn=camera2_sn)
    print(f"Stereo Camera Sn-{camera1.camera_sn}, Parameters: BaseLine-{camera1.camera.Baseline} F_pixel-{camera1.camera.f_pixel} CxCy-{camera1.camera.CxCy}")
    # print(f"Stereo Camera Sn-{camera2_sn}, Parameters: BaseLine-{camera2.camera.Baseline} F_pixel-{camera2.camera.f_pixel} CxCy-{camera2.camera.CxCy}")

    thread1 = threading.Thread(target=camera1.process_images,name="camera1_right")
    # thread2 = threading.Thread(target=camera1.run,name="camera1_plot")

    # thread2 = threading.Thread(target=camera2.process_images,name="camera2_left")


    thread1.start()
    camera1.run_app()
    # thread2.start()

    thread1.join()
    # thread2.join()

    camera1.plot_positions()
    camera1.plot_positions_and_errors()


