import numpy as np

class ExKalmanFilter:
    def __init__(self, dt=1, process_noise=1e-1, measurement_noise=1e-2, error_estimate=1):
        self.dt = dt
        # Initial state (position and velocity)
        self.x = np.zeros((2, 1))
        # State transition matrix is replaced by the state transition function f
        # Measurement matrix is replaced by the measurement function h
        self.Q = np.array([[process_noise, 0], [0, process_noise]])
        self.R = np.array([[measurement_noise]])
        self.P = np.array([[error_estimate, 0], [0, error_estimate]])

    def f(self, x):
        """ State transition function """
        # Example linear function: x_next = Ax + Bu (without control input, u)
        # Adapt this function to your system's dynamics
        A = np.array([[1, self.dt], [0, 1]])
        return np.dot(A, x)

    def h(self, x):
        """ Measurement function """
        # Example linear measurement: z = Hx
        # Adapt this function to your system's measurement model
        H = np.array([[1, 0]])
        return np.dot(H, x)

    def F_jacobian(self, x):
        """ Jacobian of the state transition function """
        # This should be derived based on your specific f
        return np.array([[1, self.dt], [0, 1]])

    def H_jacobian(self, x):
        """ Jacobian of the measurement function """
        # This should be derived based on your specific h
        return np.array([[1, 0]])

    def predict(self):
        # Use the state transition function instead of matrix
        self.x = self.f(self.x)
        F_jac = self.F_jacobian(self.x)
        self.P = np.dot(F_jac, np.dot(self.P, F_jac.T)) + self.Q

    def update(self, z):
        z_pred = self.h(self.x)
        y = z - z_pred
        H_jac = self.H_jacobian(self.x)
        S = np.dot(H_jac, np.dot(self.P, H_jac.T)) + self.R
        K = np.dot(np.dot(self.P, H_jac.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(H_jac, self.P))

    def predict_n_steps_ahead(self, n=10):
        # Temporarily copy the current state and covariance
        x_temp = np.copy(self.x)
        P_temp = np.copy(self.P)
        
        # List to hold predicted states
        predictions = []
        
        for _ in range(n):
            # Predict using the state transition function
            x_temp = self.f(x_temp)
            
            # Optionally, update the error covariance matrix here if needed
            F_jac = self.F_jacobian(x_temp)
            P_temp = np.dot(F_jac, np.dot(P_temp, F_jac.T)) + self.Q
            
            # Store the predicted state
            predictions.append(x_temp.flatten())
        
        return predictions


