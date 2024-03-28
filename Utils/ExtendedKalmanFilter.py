import numpy as np

class ExKalmanFilter:
    def __init__(self,dt=1/60, process_noise=0.09, measurement_noise=5.5, error_estimate=1):
        self.dt = dt
        # Initial state (x_position, y_position, x_velocity, y_velocity)
        self.x = np.zeros((4, 1))
        # Process noise covariance matrix for a 4D state vector
        self.Q = process_noise * np.eye(4)
        # Measurement noise covariance matrix (assuming single-dimensional measurement, e.g., x_position)
        self.R = np.array([[measurement_noise]])
        # Error covariance matrix for a 4D state vector
        self.P = error_estimate * np.eye(4)

    def f(self, x):
        """ State transition function including gravity """
        g = -9.81  # Gravity, m/s^2
        A = np.array([[1, 0, self.dt, 0],  # Transition for x_position
                      [0, 1, 0, self.dt],  # Transition for y_position
                      [0, 0, 1, 0],       # Transition for x_velocity
                      [0, 0, 0, 1]])      # Transition for y_velocity
        B = np.array([0, 0.5 * self.dt**2, 0, self.dt]).reshape(-1, 1)  # Control input for gravity affects y_velocity
        u = g
        new_x = np.dot(A, x) + B * u
        return new_x

    def h(self, x):
        """ Measurement function """
        # If the measurement includes both x_position and y_position
        H = np.array([[1, 0, 0, 0], 
                      [0, 1, 0, 0]])
        return np.dot(H, x)

    def F_jacobian(self, x):
        """ Jacobian of the state transition function """
        # This should be consistent with your specific f
        return np.array([[1, 0, self.dt, 0],
                         [0, 1, 0, self.dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def H_jacobian(self, x):
        """ Jacobian of the measurement function """
        # This should be consistent with your specific h
        # Adjust according to the dimensions of your measurement vector
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

    def predict(self):
        # Use the state transition function instead of matrix
        self.x = self.f(self.x)
        F_jac = self.F_jacobian(self.x)
        self.P = np.dot(F_jac, np.dot(self.P, F_jac.T)) + self.Q

    def update(self, z):
        # Assuming z is a 2D measurement (x_position, y_position)
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
            P_temp