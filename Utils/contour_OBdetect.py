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
from scipy.optimize import fsolve
from pyqtgraph.opengl import GLMeshItem


class SignalEmitter(QObject):
    """Emit signals with new data."""
    newData = pyqtSignal(np.ndarray, np.ndarray, np.ndarray,np.ndarray)


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
            dt = 1  # Time step (seconds) - adjust based on your measurement frequency
            measurement_noise = 0.05  # Measurement noise variance (for each x, y, z)
            error_estimate = 0.1  # Initial error estimate
            process_noise_pos=1.9
            process_noise_vel=0.5
            process_noise_acc=0.5
            self.ekf = ExtendedKalmanFilter(dt,  process_noise_pos, process_noise_vel, process_noise_acc, measurement_noise, error_estimate)

            self.left=balltracking()
            self.right=balltracking()
            self.actual_positions = []  # List to store (X, Y, Z) tuples of actual positions
            self.futurepredicted_positions=[]
            self.parabolic_pred=[]
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

        
        w.setCameraPosition(distance=500)
        w.setBackgroundColor('grey')

        grid = gl.GLGridItem()
        grid.scale(10, 10, 10)
        w.addItem(grid)
    # Define the vertices and faces of the plane
        vertices = np.array([
            [60, 0, 0],  # Point 1
            [0, 60, 0],  # Point 2
            [0, 60, 60],  # Point 3
            [60, 0, 60]   # Point 4
        ])
        faces = np.array([
            [0, 1, 2],  # First triangle
            [0, 2, 3]   # Second triangle
        ])
        
        # Define the color for each vertex
        colors = np.array([
            [1, 0, 0, 0.3],  # Red, semi-transparent
            [1, 0, 0, 0.3],  # Green, semi-transparent
            [1, 0, 0, 0.3],  # Blue, semi-transparent
            [1, 0, 0, 0.3]   # Yellow, semi-transparent
        ])
        
        # Create the mesh item
        plane_mesh = GLMeshItem(vertexes=vertices, faces=faces, faceColors=colors, drawEdges=True)
        w.addItem(plane_mesh)

        self.actualScatter = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=10)
        w.addItem(self.actualScatter)

        self.predictedScatter = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(0, 1, 0, 1), size=10)
        w.addItem(self.predictedScatter)

        self.parabolic_Scatter = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(0, 0, 1, 1), size=10)
        w.addItem(self.parabolic_Scatter)

        self.intersect = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 1, 0, 1), size=10)
        w.addItem(self.intersect)
        axis = gl.GLAxisItem()
        axis.setSize(150, 150, 150)
        w.addItem(axis)

        w.show()
        return w
    
    def update_plot(self, actual, predicted,trajectory,intersect):
        """Update plot with new data."""
        self.actualScatter.setData(pos=actual, color=(1, 0, 0, 1), size=5)
        self.predictedScatter.setData(pos=predicted, color=(0, 1, 0, 1), size=5)
        self.parabolic_Scatter.setData(pos=trajectory, color=(0, 0, 1, 1), size=5)    
        # Update intersection plot only if there are intersection points
        if intersect.size > 0:
            self.intersect.setData(pos=intersect, color=(1, 1, 0, 1), size=15)
        else:
            self.intersect.setData(pos=np.empty((0, 3)))  

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
                self.parabolic_pred.clear()
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
        it=[]
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
            

            
            # def is_point_on_plane(point, vertices):
            #     normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            #     D = -np.dot(normal, vertices[0])
            #     return np.isclose(np.dot(normal, point) + D, 0)

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
                # Assuming your plane is defined by the following vertices
                vertices = np.array([[10, 0, 0], [0, 10, 0], [0, 10, 10], [10, 0, 10]])
                normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                point_on_plane = vertices[0]
                D = -np.dot(normal, point_on_plane)
                
                # Update your method where you process the EKF output
                nt=self.ekf.predict_n_steps_ahead(self.N_prediction)
                for n in nt:
                    self.futurepredicted_positions.append([n[0], n[1], n[2]])
                
                
                self.actual_positions.append([X, Y, Z])
                intersection_point=[]
                if intersection_point is not None:
                    self.emitter.newData.emit(
                        np.array(self.actual_positions),
                        np.array(self.futurepredicted_positions),
                        np.array(self.parabolic_pred),
                        np.array([intersection_point])
                    )
                else:
                    self.emitter.newData.emit(
                        np.array(self.actual_positions),
                        np.array(self.futurepredicted_positions),
                        np.array(self.parabolic_pred),
                        np.array([])  # Emit an empty array if there's no intersection
                    )
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
        # print(self.futurepredicted_positions)

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

