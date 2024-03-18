
import os
import sys
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils'))

# Add the 'utils' folder to sys.path
sys.path.append(utils_path)
import cv2
import torch
import keyboard
import numpy as np
from time import sleep
from ultralytics import YOLO
from CamParam import *
# from threading import Lock, Thread
from pose_esitmation import pose
from StereoCamera import *

# Serial Numbers
left_cam_sn="SN20778657"
camera_sn="25062645"


# select parameters
resolution={"2K":["2k",4416,1242],"FHD":["FHD",3840,1080],"HD":["HD",2560,720],"VGA":["VGA",1344,376]} # 2K, FHD, HD, VGD
metrics="cm" # cm,m,ft
f_pixel=None  # Focal length in pixels
baseline = None  # Baseline in meters
box_points={}


# set parameters
set_resolution = resolution["HD"]
camera=Camera(camera_id=1,camera_sn=camera_sn)
baseline,f_pixel,CxCy=camera.Baseline,camera.f_pixel,camera.CxCy
print(baseline,f_pixel,CxCy)
def main() :
    # Check if a GPU is available and select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Load the YOLO model and move it to the selected device
    model = YOLO('PoseEstimation\model\yolov8l_tennis_ball.pt').to(device)
    # print(model.predictor)
    # Open the ZED camera or stereo camera using OpenCV
    print(cv2.VideoCapture(1))
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error opening video stream")
        exit(-1)

    # Set the video resolution to HD720 (2560*720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, set_resolution[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, set_resolution[2])

    while not keyboard.is_pressed('q'):
        # Get a new frame from camera
        retval, frame = cap.read()
        if not retval:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Extract left and right images from side-by-sidel
        left_image, right_image = np.split(frame, 2, axis=1)

        # Process each image for object detection
        for img, label in zip([left_image, right_image], ["Left", "Right"]):
            results = model(img)
            if len(results) > 0:
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.to(device)
                        confidences = result.boxes.conf.to(device)
                        class_ids = result.boxes.cls.to(device)

                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                            box_points[label] = {"box": (int(x1), int(y1), int(x2), int(y2)), "center": ((int(x1) + int(x2)) / 2, (int(y1) + int(y2)) / 2)}

                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(img, f"ID: {class_ids[i]}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        if len(results) > 0 and  "Right" in box_points and "Left" in box_points:
            X,Y,depth = pose(box_points["Right"]["center"], box_points["Left"]["center"], baseline, f_pixel,CxCy)
            for img_label in ["Left", "Right"]:
                center_x, center_y = box_points[img_label]["center"]
                img = left_image if img_label == "Left" else right_image
                cv2.putText(img, f"X: {X:.2f}{metrics}", (int(center_x), int(center_y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.putText(img, f"Y: {Y:.2f}{metrics}", (int(center_x), int(center_y + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.putText(img, f"Depth: {depth:.2f}{metrics}", (int(center_x), int(center_y + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                cv2.imshow(img_label, img)

                        
            else:
                print("No detections")
            # sleep(0.5)
            
        if cv2.waitKey(30) >= 0:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()