import time
import cv2
import keyboard
from StereoCamera import *
from pose_esitmation import *
from ball_tracking import *
import matplotlib.pyplot as plt
from ExtendedKalmanFilter import *
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal
import threading
import time
import pyqtgraph.opengl as gl
from PyQt5.QtGui import QQuaternion


class SignalEmitter(QObject):
    """Emit signals with new data."""
    newData = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)


class contour_OB:
    def __init__(self,camera_id,camera_sn,thershold=0.9,metric='cm',N_prediction=15,cam_state=False):
            self.cam_state=cam_state
            self.N_prediction=N_prediction
            self.model_thershold=thershold
            self.camera=Camera(camera_id=camera_id,camera_sn=camera_sn,resolution="HD",desired_fps=60,cam=self.cam_state)
            self.metric=metric
            self.camera_sn=camera_sn
            self.pose=[]



            # Initialize the EKF with your specific parameters
            dt = 1/60  # Time step (seconds) - adjust based on your measurement frequency
            process_noise = 0.1  # Process noise variance
            measurement_noise = 5.5  # Measurement noise variance (for each x, y, z)
            error_estimate = 1  # Initial error estimate
            self.ekf = ExKalmanFilter(dt, process_noise, measurement_noise, error_estimate)

            self.left=balltracking()
            self.right=balltracking()
            self.actual_positions = []  # List to store (X, Y, Z) tuples of actual positions
            self.futurepredicted_positions=[]
            self.parbolic_pred=[]
            self.magnitude_list=[]
            ####################################################################################
            self.app = QApplication(sys.argv)
            self.window = self.setup_ui()
            self.actualData = []
            self.predictedData = []
            self.emitter = SignalEmitter()
            self.emitter.newData.connect(self.update_plot)


    def setup_ui(self):
        """Set up the GUI."""
        w = gl.GLViewWidget()
        w.setWindowTitle('Real-time 3D Plotting of Actual vs. Predicted Points')
        # rotation = QQuaternion.fromEulerAngles(90 ,45,0)

        
        w.setCameraPosition(distance=500)
        w.setBackgroundColor('grey')

        grid = gl.GLGridItem()
        grid.scale(10, 10, 10)
        w.addItem(grid)

        self.actualScatter = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=10)
        w.addItem(self.actualScatter)

        self.predictedScatter = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(0, 1, 0, 1), size=10)
        w.addItem(self.predictedScatter)

        self.parabolic_Scatter = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(0, 0, 1, 1), size=10)
        w.addItem(self.parabolic_Scatter)

        axis = gl.GLAxisItem()
        axis.setSize(150, 150, 150)
        w.addItem(axis)

        w.show()
        return w
    
    def update_plot(self, actual, predicted,trajectory):
        """Update plot with new data."""
        self.actualScatter.setData(pos=actual, color=(1, 0, 0, 1), size=5)
        self.predictedScatter.setData(pos=predicted, color=(0, 1, 0, 1), size=5)
        self.parabolic_Scatter.setData(pos=trajectory, color=(0, 0, 1, 1), size=5)    

    def run_app(self):
        self.app.exec_()


    def process_images(self):
        fps_counter = 0
        start_time = time.time()
        fps=0
        while not keyboard.is_pressed("q"):
            if keyboard.is_pressed("r"):
                self.actual_positions.clear()
                self.futurepredicted_positions.clear()
                self.parbolic_pred.clear()
                self.magnitude_list.clear()
                self.actualData.clear()
                self.predictedData.clear()
                if self.actualScatter:
                    self.actualScatter.setData(pos=np.empty((0, 3)))
                if self.predictedScatter:
                    self.predictedScatter.setData(pos=np.empty((0, 3)))
                if self.parabolic_Scatter:
                    self.parabolic_Scatter.setData(pos=np.empty((0, 3)))
            
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
                self.ekf.dt=(1/(fps+1))

                # get ball detection on left and  right frame #
                left_image,Lcenter=self.left.pos(left_image)
                right_image,Rcenter=self.right.pos(right_image)
                
                cv2.putText(right_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(left_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.calculate_position(Lcenter,Rcenter, left_image, right_image)
               
        self.camera.release()


    def calculate_position(self, Lcenter,Rcenter, left_image, right_image):
        magnitude=0
        v_initial = np.array([0, 0, 0])
        if right_image is not None and left_image is not None and Lcenter is not None and Rcenter is not None   :
            X, Y, Z = pose(Rcenter,Lcenter, self.camera.Baseline, self.camera.f_pixel, self.camera.CxCy)

            if X is None or Y is None or Z is None:
                print("Skipping pose calculation due to invalid disparity.")
                return

            self.ekf.update(np.array([X,Y,Z]))
            self.ekf.predict()
            next_position1 = self.ekf.x[:3].flatten()
            next_velocity = self.ekf.x[3:].flatten()
            magnitude = np.linalg.norm(next_velocity)
            self.magnitude_list.append(magnitude)
            predicted_X = next_position1[0]
            predicted_Y = next_position1[1]
            predicted_Z = next_position1[2]
            

            


            if not np.isnan(predicted_X) and not np.isnan(predicted_Y) and not np.isinf(predicted_X) and not np.isinf(predicted_Y):
                # self.futurepredicted_positions.clear()
                # Estimate initial velocity based on the first two points (or more if you like)
                dt =self.ekf.dt # Time difference between points, adjust as necessary

                # Constants
                g = 9.81  # Gravity (m/s^2)

                # Time for future points
                time_steps = np.linspace(0, 2, num=self.N_prediction)  # Adjust the range and number of points as necessary

                # Initialize arrays for future trajectory points
                future_points = np.zeros((len(time_steps), 3))

                

                for i in range(self.N_prediction):

                    self.ekf.update(np.array([X,Y,Z]))

                    # Predict next position
                    self.ekf.predict()
                    next_position = self.ekf.x[:3].flatten()

                    
                    self.futurepredicted_positions.append([next_position[0], next_position[1], next_position[2]])
                    
                    if len(self.futurepredicted_positions) >= self.N_prediction:
                        # Convert the last two positions from lists to NumPy arrays
                        pos1 = np.array(self.futurepredicted_positions[-1])
                        pos2 = np.array(self.futurepredicted_positions[int((-self.N_prediction)/2)])

                        # Calculate initial velocity as a NumPy array
                        v_initial = (pos1 - pos2) / dt
                        
                        # Calculate future points based on estimated initial velocity and gravity
                        future_points = []
                        for t in time_steps:
                            next_pos = pos1 + v_initial * t + 0.5 * np.array([0, 0, -g]) * t**2
                            future_points.append(next_pos) 
                        self.parbolic_pred=future_points

                # Ensure img is defined and refers to a valid image
                self.actual_positions.append([X, Y, Z])
                
                self.emitter.newData.emit(np.array(self.actual_positions), np.array(self.futurepredicted_positions),np.array(self.parbolic_pred))
            else:
                
                print("Skipping drawing circle due to invalid predicted_X or predicted_Y values.")
            ##left image
            
            cv2.putText(left_image, f"X: {X:.2f}", (int(Lcenter[0]), int(Lcenter[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(left_image, f"Y: {Y:.2f}", (int(Lcenter[0]), int(Lcenter[1] + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(left_image, f"Z: {Z:.2f}", (int(Lcenter[0]), int(Lcenter[1] + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            
            ##right image
            cv2.putText(right_image, f"X: {X:.2f}", (int(Rcenter[0]), int(Rcenter[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(right_image, f"Y: {Y:.2f}", (int(Rcenter[0]), int(Rcenter[1] + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.putText(right_image, f"Z: {Z:.2f}", (int(Rcenter[0]), int(Rcenter[1] + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            
        
        cv2.putText(left_image, f"magnitude: {magnitude:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(right_image, f"magnitude: {magnitude:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow(f"{self.camera_sn}Right", right_image)
        cv2.imshow(f"{self.camera_sn}Left", left_image)
        cv2.waitKey(1)
        
    def magnitude_plot(self):
        plt.figure(figsize=(10, 6))  # Create a figure with a custom size
        plt.plot(self.magnitude_list, marker='o', linestyle='-', color='b')  # Plot magnitude with blue line and circle markers
        plt.title('Magnitude of Velocity Over Time')  # Title of the plot
        plt.xlabel('Time Step')  # X-axis label
        plt.ylabel('Magnitude of Velocity (m/s)')  # Y-axis label
        plt.grid(True)  # Show grid lines for better readability
        plt.show()



    def plot_positions_and_errors(self):
        if not self.actual_positions or not self.futurepredicted_positions:
            print("Not enough data for error calculation.")
            return

        # Reset errors list for each calculation
        merrors = []
        temp=None
        for i in range(len(self.actual_positions)-1):  # Subtract 1 to avoid index out of bounds
            actual = self.actual_positions[i+1]
            # Calculate Euclidean distance between actual and each predicted position
            # errors = [np.linalg.norm(np.array(actual) - np.array(pred)) for pred in self.futurepredicted_positions]
            for j in range(self.N_prediction):
                err = np.linalg.norm(np.array(actual) - np.array(self.futurepredicted_positions[j]))
                if temp==None or err<temp:
                    # print("error",actual,self.futurepredicted_positions[j],e)
                    temp=err
                self.futurepredicted_positions.pop(0)
            # Find the minimum error for this actual position
                merrors.append(temp)

        # Plot Prediction Errors
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        plt.plot(merrors, label='Prediction Error', marker='o', linestyle='-')
        plt.xlabel('Measurement Number')
        plt.ylabel('Error (Euclidean distance)')
        plt.title(f'Error between Actual and Predicted Future Trajectories')
        plt.legend()
        plt.grid(True)

        # Assuming self.magnitude_list is populated elsewhere in your class
        if hasattr(self, 'magnitude_list') and self.magnitude_list:
            # Plot Magnitude of Velocity Over Time
            plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
            plt.plot(self.magnitude_list, marker='o', linestyle='-', color='b')
            plt.title('Magnitude of Velocity Over Time')
            plt.xlabel('Time Step')
            plt.ylabel('Magnitude of Velocity (m/s)')
            plt.grid(True)

        plt.tight_layout()  # Adjust layout to not overlap
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

