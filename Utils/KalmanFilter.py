import numpy as np

class kalmanfilter:
    def __init__(self, dt=1, process_noise=1e-1, measurement_noise=1e-2, error_estimate=1):
        self.dt = dt  # Time step
        # State transition matrix
        self.A = np.array([[1, dt], [0, 1]])
        # Measurement matrix
        self.H = np.array([[1, 0]])
        # Process noise covariance
        self.Q = np.array([[process_noise, 0], [0, process_noise]])
        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])
        # Error covariance matrix
        self.P = np.array([[error_estimate, 0], [0, error_estimate]])
        # Initial state (position and velocity)
        self.x = np.zeros((2, 1))
    
    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
    
    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))