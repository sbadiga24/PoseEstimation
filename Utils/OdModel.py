import time
import cv2
import numpy as np
import torch
import keyboard
from ultralytics import YOLO
from StereoCamera import *
from pose_esitmation import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ExtendedKalmanFilter import ExKalmanFilter


class ObjectDetection:
    def __init__(self,model_path,camera_id,camera_sn,thershold=0.9,metric='cm'):
        self.model_path=model_path
        self.model_thershold=thershold
        self.camera=Camera(camera_id=camera_id,camera_sn=camera_sn,resolution="HD")
        self.metric=metric
        self.camera_sn=camera_sn
        self.pose=[]
        self.model=self.load_model()
        self.kf_x = ExKalmanFilter()
        self.kf_y = ExKalmanFilter()
        self.kf_z = ExKalmanFilter()
        self.actual_positions = []  # List to store (X, Y, depth) tuples of actual positions
        self.predicted_positions = []  # List to store (X, Y, depth) tuples of predicted positions
    

    

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Model Running on device:", device)
        return YOLO(self.model_path).to(device)
    
    def process_images(self):
        fps_counter = 0
        start_time = time.time()

        while not keyboard.is_pressed('q'):
            fps_counter += 1
            left_image, right_image = self.camera.capture_frame()
            box_points = {}
            images_to_show = {'Left': left_image, 'Right': right_image}  # Prepare images for display

            for img, label in zip([left_image, right_image], ["Left", "Right"]):
                results = self.model(img, self.model_thershold)
                results_list = list(results)  # Convert generator to list
                if len(results_list) > 0:
                    for result in results_list:
                        if result.boxes is not None and len(result.boxes) > 0:
                            boxes = result.boxes.xyxy.to(self.model.device)
                            class_ids = result.boxes.cls.to(self.model.device)
                            for i, box in enumerate(boxes):
                                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                                box_points[label] = {"box": (int(x1), int(y1), int(x2), int(y2)), "center": ((int(x1) + int(x2)) / 2, (int(y1) + int(y2)) / 2)}
                                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(img, f"ID: {class_ids[i]}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    if len(results_list) > 0 and "Right" in box_points and "Left" in box_points:
                        self.calculate_position(box_points, left_image, right_image)
                        # self.plot_positions()
                images_to_show[label] = img
             # Calculate and display FPS every second
            if time.time() - start_time >= 1:
                fps = fps_counter / (time.time() - start_time)
                # print(f"FPS: {fps:.2f}")
                fps_counter = 0
                start_time = time.time()
            # Display both images outside the conditional blocks
            for label, img in images_to_show.items():
                cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.imshow(f"{self.camera_sn},{label}", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break            
        self.camera.release()            


    def calculate_position(self, box_points, left_image, right_image):
        X, Y, depth = pose(box_points["Right"]["center"], box_points["Left"]["center"], self.camera.Baseline, self.camera.f_pixel, self.camera.CxCy, self.metric)
        if X is None or Y is None or depth is None:
            print("Skipping pose calculation due to invalid disparity.")
            return
        
        # Update Kalman Filter with new measurements
        self.kf_x.update(np.array([[X]]))
        self.kf_y.update(np.array([[Y]]))
        self.kf_z.update(np.array([[depth]]))
        
        # Predict next position
        self.kf_x.predict()
        self.kf_y.predict()
        self.kf_z.predict()
        
        # Use predicted values for display
        predicted_X = self.kf_x.x[0, 0]
        predicted_Y = self.kf_y.x[0, 0]
        predicted_depth = self.kf_z.x[0, 0]

        for img_label in ["Left", "Right"]:
            center_x, center_y = box_points[img_label]["center"]
            img = left_image if img_label == "Left" else right_image
            cv2.putText(img, f"X: {X:.2f}{self.metric}", (int(center_x), int(center_y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(img, f"Y: {Y:.2f}{self.metric}", (int(center_x), int(center_y + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(img, f"Depth: {depth:.2f}{self.metric}", (int(center_x), int(center_y + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            if not np.isnan(predicted_X) and not np.isnan(predicted_Y) and not np.isinf(predicted_X) and not np.isinf(predicted_Y):
                predicted_X_int = int(predicted_X)
                predicted_Y_int = int(predicted_Y)
                
                # Ensure img is defined and refers to a valid image
                img = left_image if 'Left' in box_points else right_image
                
                # Draw the circle with corrected integer coordinates
                # cv2.circle(img, (predicted_X_int, predicted_Y_int), 15, (0, 225, 0), -1)
                # cv2.putText(img, f"X: {predicted_X:.2f}{self.metric}", (int(predicted_X), int(predicted_Y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (222, 222, 12), 2)
                # cv2.putText(img, f"Y: {predicted_Y:.2f}{self.metric}", (int(predicted_X), int(predicted_Y + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (222, 222, 12), 2)
                # cv2.putText(img, f"Depth: {predicted_depth:.2f}{self.metric}", (int(predicted_X), int(predicted_Y + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (222, 222, 12), 2)
                self.actual_positions.append((X, Y, depth))
                self.predicted_positions.append((predicted_X, predicted_Y, predicted_depth))
            else:
                print("Skipping drawing circle due to invalid predicted_X or predicted_Y values.")

    def plot_positions(self):
        # Separate actual and predicted positions into X, Y, Z components for plotting
        actual_x, actual_y, actual_z = zip(*self.actual_positions)
        predicted_x, predicted_y, predicted_z = zip(*self.predicted_positions)
        
        # Create 3D plot
        fig = plt.figure(f'{self.camera_sn}')
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot actual positions
        ax.scatter(actual_x, actual_y, actual_z, c='blue', marker='o', label='Actual')
        # Plot predicted positions
        ax.scatter(predicted_x, predicted_y, predicted_z, c='red', marker='^', label='Predicted')
        
        # Labels and legend
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.legend()
        
        plt.show()        

    def plot_positions_and_errors(self):
        # Check if there's data to plot
        if not self.actual_positions or not self.predicted_positions:
            print("No data available for plotting.")
            return
        
        # Convert lists to numpy arrays for easier manipulation
        actual = np.array(self.actual_positions)
        predicted = np.array(self.predicted_positions)
        
        # Calculate errors (Euclidean distance between actual and predicted positions)
        errors = np.sqrt(np.sum((actual - predicted)**2, axis=1))
        
        # Time or sequence axis
        t = np.arange(len(errors))
        
        # Plotting the error over time
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        plt.plot(t, errors, label='Error', marker='o')
        plt.xlabel('Measurement Number')
        plt.ylabel('Error (Euclidean distance)')
        plt.title('Error between Actual and Predicted Positions')
        plt.grid(True)
        plt.legend()
        
        actual_x, actual_y, actual_z = zip(*self.actual_positions)
        predicted_x, predicted_y, predicted_z = zip(*self.predicted_positions)
        
        # Create 3D plot
        fig = plt.figure(f'{self.camera_sn}')
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot actual positions
        ax.scatter(actual_x, actual_y, actual_z, c='blue', marker='o', label='Actual')
        # Plot predicted positions
        ax.scatter(predicted_x, predicted_y, predicted_z, c='red', marker='^', label='Predicted')
        
        # Labels and legend
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.legend()
        
        plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure area.
        plt.show()
    



    def IDL_plot_positions_and_errors(self):
        # Check if there's data to plot
        if not self.actual_positions or not self.predicted_positions:
            print("No data available for plotting.")
            return
        
        # Convert lists to numpy arrays for easier manipulation
        actual = np.array(self.actual_positions)
        predicted = np.array(self.predicted_positions)
        
        # Component-wise errors
        errors_x = np.abs(actual[:, 0] - predicted[:, 0])
        errors_y = np.abs(actual[:, 1] - predicted[:, 1])
        errors_z = np.abs(actual[:, 2] - predicted[:, 2])
        
        # Time or sequence axis
        t = np.arange(len(actual))
        
        # Plotting
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        
        # X component plot
        axs[0, 0].plot(t, actual[:, 0], 'b-', label='Actual X')
        axs[0, 0].plot(t, predicted[:, 0], 'r--', label='Predicted X')
        axs[0, 0].set_xlabel('Measurement Number')
        axs[0, 0].set_ylabel('X Position')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Error in X
        axs[0, 1].plot(t, errors_x, 'k-', label='Error in X')
        axs[0, 1].set_xlabel('Measurement Number')
        axs[0, 1].set_ylabel('Error')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # Y component plot
        axs[1, 0].plot(t, actual[:, 1], 'g-', label='Actual Y')
        axs[1, 0].plot(t, predicted[:, 1], 'r--', label='Predicted Y')
        axs[1, 0].set_xlabel('Measurement Number')
        axs[1, 0].set_ylabel('Y Position')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Error in Y
        axs[1, 1].plot(t, errors_y, 'k-', label='Error in Y')
        axs[1, 1].set_xlabel('Measurement Number')
        axs[1, 1].set_ylabel('Error')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Z component plot
        axs[2, 0].plot(t, actual[:, 2], 'c-', label='Actual Z')
        axs[2, 0].plot(t, predicted[:, 2], 'r--', label='Predicted Z')
        axs[2, 0].set_xlabel('Measurement Number')
        axs[2, 0].set_ylabel('Z Position')
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Error in Z
        axs[2, 1].plot(t, errors_z, 'k-', label='Error in Z')
        axs[2, 1].set_xlabel('Measurement Number')
        axs[2, 1].set_ylabel('Error')
        axs[2, 1].legend()
        axs[2, 1].grid(True)

        plt.tight_layout()
        plt.show()



