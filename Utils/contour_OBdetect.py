import time
import cv2
import keyboard
from StereoCamera import *
from pose_esitmation import *
from ball_tracking import *
import matplotlib.pyplot as plt
from ExtendedKalmanFilter import *


class contour_OB:
    def __init__(self,camera_id,camera_sn,thershold=0.9,metric='cm',N_prediction=15):
            self.N_prediction=N_prediction
            self.model_thershold=thershold
            camera_path="vd11_1.mp4"
            self.camera=Camera(camera_id=camera_id,camera_sn=camera_sn,resolution="HD",desired_fps=60,cam=False)
            self.metric=metric
            self.camera_sn=camera_sn
            self.pose=[]
            self.kf_x = ExKalmanFilter()
            self.kf_y = ExKalmanFilter()
            self.kf_z = ExKalmanFilter()
            self.left=balltracking()
            self.right=balltracking()
            self.actual_positions = []  # List to store (X, Y, depth) tuples of actual positions
            self.futurepredicted_positions=[]

    def process_images(self):
        fps_counter = 0
        start_time = time.time()
        fps=0
        while not keyboard.is_pressed("q"):
            left_image, right_image = self.camera.capture_frame()

            if left_image is None or right_image is None:
                print("Check camera connection")
                break
            else:
                # Calculate FPS
                current_time = time.time()
                fps_counter += 1
                if (current_time - start_time) > 1:
                    fps = fps_counter / (current_time - start_time)
                    fps_counter = 0
                    start_time = current_time

                # Display FPS on the images
                fps_text = f"FPS: {fps:.2f}"
                # get ball detection on left and  right frame #
                left_image,Lcenter=self.left.pos(left_image)
                right_image,Rcenter=self.right.pos(right_image)
                
                cv2.putText(right_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(left_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.calculate_position(Lcenter,Rcenter, left_image, right_image)
               
        self.camera.release()


    def calculate_position(self, Lcenter,Rcenter, left_image, right_image):
        if right_image is not None and left_image is not None and Lcenter is not None and Rcenter is not None   :
            X, Y, depth = pose(Rcenter,Lcenter, self.camera.Baseline, self.camera.f_pixel, self.camera.CxCy)
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
            



            if not np.isnan(predicted_X) and not np.isnan(predicted_Y) and not np.isinf(predicted_X) and not np.isinf(predicted_Y):
                # self.futurepredicted_positions.clear()

                for i in range(self.N_prediction):
                    self.kf_x.update(np.array([[X]]))
                    self.kf_y.update(np.array([[Y]]))
                    self.kf_z.update(np.array([[depth]]))
                        # Predict next position
                    self.kf_x.predict()
                    self.kf_y.predict()
                    self.kf_z.predict()
                
                    # Use predicted values for display
                    fpredicted_X = np.round(self.kf_x.x[0, 0], 2)
                    fpredicted_Y = np.round(self.kf_y.x[0, 0], 2)
                    fpredicted_depth = np.round(self.kf_z.x[0, 0], 2)
                    self.futurepredicted_positions.append([fpredicted_X, fpredicted_Y, fpredicted_depth])
                    

                # Ensure img is defined and refers to a valid image
                self.actual_positions.append([X, Y, depth])
            else:
                print("Skipping drawing circle due to invalid predicted_X or predicted_Y values.")
            ##left image
            
            cv2.putText(left_image, f"X: {X:.2f}", (int(Lcenter[0]), int(Lcenter[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(left_image, f"Y: {Y:.2f}", (int(Lcenter[0]), int(Lcenter[1] + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(left_image, f"Depth: {depth:.2f}", (int(Lcenter[0]), int(Lcenter[1] + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            ##right image
            cv2.putText(right_image, f"X: {X:.2f}", (int(Rcenter[0]), int(Rcenter[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(right_image, f"Y: {Y:.2f}", (int(Rcenter[0]), int(Rcenter[1] + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(right_image, f"Depth: {depth:.2f}", (int(Rcenter[0]), int(Rcenter[1] + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            


        cv2.imshow(f"{self.camera_sn}Right", right_image)
        cv2.imshow(f"{self.camera_sn}Left", left_image)
        cv2.waitKey(1)
        

    def plot_positions_and_errors(self):
        # Ensure there are actual positions and predicted future positions to calculate errors
        if not self.actual_positions or not self.futurepredicted_positions:
            print("Not enough data for error calculation.")
            return
        # print("act\n",self.actual_positions)

        # print(self.futurepredicted_positions[1+8])
        # Calculate errors
        errors = []
        for i in range(len(self.actual_positions)-1):
            actual=self.actual_positions[i+1]
            first_predicted=self.futurepredicted_positions[self.N_prediction]
            for j in range(self.N_prediction):
                self.futurepredicted_positions.pop(j)
            # first_predicted = predicted_set[8]  # Assuming the first prediction corresponds to the actual point
            error = np.linalg.norm(np.array(actual) - np.array(first_predicted))
            errors.append(error)
        
        # Plot errors
        plt.figure()
        plt.plot(errors, label='Prediction Error', marker='o')
        plt.xlabel('Measurement Number')
        plt.ylabel('Error (Euclidean distance)')
        plt.title(f'Error between Actual and Predicted Future Trajectories in "{self.metric}"')
        plt.legend()
        plt.show()


    def plot_positions(self):
        # Separate actual and predicted positions into X, Y, Z components for plotting

        actual_x, actual_y, actual_z = zip(*self.actual_positions)
        fpredicted_x, fpredicted_y, fpredicted_z = zip(*self.futurepredicted_positions)

        
        # Create 3D plot
        fig = plt.figure(f'{self.camera_sn}')
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot actual positions
        ax.plot(actual_x, actual_y, actual_z, c='blue',marker='o', label='Actual', alpha=0.2)
        # Plot predicted positions
        ax.plot(fpredicted_x, fpredicted_y, fpredicted_z, c='red',marker='>', label='fPredicted', alpha=0.2)
        
        # Labels and legend
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.legend()
        
        plt.show() 

