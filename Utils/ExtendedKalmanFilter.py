import numpy as np

import numpy as np

class ExKalmanFilter:
    def __init__(self, dt=1/60, process_noise=0.1, measurement_noise=5.5, error_estimate=1):
        self.dt = dt
        # Extend initial state to include z_position and z_velocity
        self.x = np.zeros((6, 1))  # [x_position, y_position, z_position, x_velocity, y_velocity, z_velocity]
        # Adjust Q, R, and P for the extended state
        self.Q = process_noise * np.eye(6)  # Process noise for 6D state
        self.R = measurement_noise * np.eye(3)  # Assuming 3D measurements (x, y, z positions)
        self.P = error_estimate * np.eye(6)  # Error estimate for 6D state

    def f(self, x):
        """ State transition function including gravity """
        g = -9.81  # Gravity affects z_velocity
        A = np.array([[1, 0, 0, self.dt, 0, 0],       # x_position
                      [0, 1, 0, 0, self.dt, 0],       # y_position
                      [0, 0, 1, 0, 0, self.dt],       # z_position
                      [0, 0, 0, 1, 0, 0],             # x_velocity
                      [0, 0, 0, 0, 1, 0],             # y_velocity
                      [0, 0, 0, 0, 0, 1]])            # z_velocity
        B = np.array([0, 0, 0.5 * self.dt**2, 0, 0, self.dt]).reshape(-1, 1)  # Gravity only affects z_velocity
        u = g
        new_x = np.dot(A, x) + B * u
        return new_x

    def h(self, x):
        """ Measurement function for 3D position """
        H = np.array([[1, 0, 0, 0, 0, 0],  # x_position
                      [0, 1, 0, 0, 0, 0],  # y_position
                      [0, 0, 1, 0, 0, 0]]) # z_position
        return np.dot(H, x)

    def F_jacobian(self, x):
        """ Jacobian of the state transition function """
        # The Jacobian matrix remains the same since f is linear in x
        return np.array([[1, 0, 0, self.dt, 0, 0],
                         [0, 1, 0, 0, self.dt, 0],
                         [0, 0, 1, 0, 0, self.dt],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])

    def H_jacobian(self, x):
        """ Jacobian of the measurement function """
        # Adjusted for 3D measurement
        return np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])

    # The predict and update methods remain largely the same
    def predict(self):
        self.x = self.f(self.x)
        F_jac = self.F_jacobian(self.x)
        self.P = np.dot(F_jac, np.dot(self.P, F_jac.T)) + self.Q

    def update(self, z):
        z_pred = self.h(self.x)
        y = z.reshape(-1, 1) - z_pred
        H_jac = self.H_jacobian(self.x)
        S = np.dot(H_jac, np.dot(self.P, H_jac.T)) + self.R
        K = np.dot(np.dot(self.P, H_jac.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(H_jac, self.P))

    def predict_n_steps_ahead(self, n=10):
        x_temp = np.copy(self.x)
        predictions = []
        for _ in range(n):
            x_temp = self.f(x_temp)
            predictions.append(x_temp[:3].flatten())  # Extract and store only position data
        return predictions

# Usage remains similar; you'd now provide z measurements in your update calls.
