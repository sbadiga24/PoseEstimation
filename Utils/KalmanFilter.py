#https://pysource.com/2021/10/29/kalman-filter-predict-the-trajectory-of-an-object/
import cv2
import numpy as np


class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)  # Changed to 6 dynamic params (x, y, z, dx, dy, dz), 3 measured params (x, y, z)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], 
                                              [0, 1, 0, 0, 0, 0], 
                                              [0, 0, 1, 0, 0, 0]], np.float32)

        self.kf.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0], 
                                             [0, 1, 0, 0, 1, 0], 
                                             [0, 0, 1, 0, 0, 1],
                                             [0, 0, 0, 1, 0, 0], 
                                             [0, 0, 0, 0, 1, 0], 
                                             [0, 0, 0, 0, 0, 1]], np.float32)

        # Initialize other necessary matrices (process noise covariance, measurement noise covariance, etc.) here

    def predict(self, coordX, coordY, coordZ):
        '''This function estimates the position of the object in 3D space'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)], [np.float32(coordZ)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y, z = int(predicted[0]), int(predicted[1]), int(predicted[2])
        return x, y, z
