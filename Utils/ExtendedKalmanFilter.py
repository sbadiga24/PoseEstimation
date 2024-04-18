
import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, dt=1/60, process_noise_pos=0.1, process_noise_vel=0.1, process_noise_acc=0.1, measurement_noise=0.001, error_estimate=1):
        self.dt = dt
        self.x = np.zeros((9, 1))  # [x, y, z, vx, vy, vz, ax, ay, az]
        self.Q = np.eye(9) * np.array([process_noise_pos, process_noise_pos, process_noise_pos,
                                       process_noise_vel, process_noise_vel, process_noise_vel,
                                       process_noise_acc, process_noise_acc, process_noise_acc])  # Process noise
        self.R = measurement_noise * np.eye(3)  # Measurement noise for position only
        self.P = error_estimate * np.eye(9)  # Initial estimate error covariance

    def F_jacobian(self):
        """ Jacobian of the state transition function, should be a 9x9 matrix """
        F = np.eye(9)
        F[0:3, 3:6] = self.dt * np.eye(3)  # Position update depends on velocity
        F[3:6, 6:9] = self.dt * np.eye(3)  # Velocity update depends on acceleration
        return F

    def f(self, x):
        """ State transition function """
        A = self.F_jacobian()
        new_x = np.dot(A, x)
        # Account for constant acceleration in the model (e.g., gravity, minus drag)
        new_x[5] += self.dt * (-9.81)  # Gravity effect on z-axis acceleration
        return new_x

    def h(self, x):
        """ Measurement function for 3D position """
        H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],  # x_position
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],  # y_position
                      [0, 0, 1, 0, 0, 0, 0, 0, 0]]) # z_position
        return np.dot(H, x)

    def H_jacobian(self):
        """ Jacobian of the measurement function """
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0]])

    def predict(self):
        """ Prediction step using the state transition function and Jacobian """
        F = self.F_jacobian()
        self.x = self.f(self.x)
        self.P = np.dot(F, np.dot(self.P, F.T)) + self.Q

    def update(self, z):
        """ Update step with measurement z """
        z_pred = self.h(self.x)
        y = z.reshape(-1, 1) - z_pred
        H = self.H_jacobian()
        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(H, self.P))

    def predict_n_steps_ahead(self, n=10):
        x_temp = np.copy(self.x)
        print(x_temp)
        predictions = []
        for _ in range(n):
            x_temp = self.f(x_temp)
            predictions.append(x_temp.flatten())  # Extract and store the entire state
        return predictions


