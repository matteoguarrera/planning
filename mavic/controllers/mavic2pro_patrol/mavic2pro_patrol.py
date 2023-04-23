# Copyright 1996-2022 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example of Python controller for Mavic patrolling around the house.
   Open the robot window to see the camera view.
   This demonstrates how to go to specific world coordinates using its GPS, imu and gyroscope.
   The drone reaches a given altitude and patrols from waypoint to waypoint."""


# /usr/local/bin/webots

from controller import Robot
from datetime import datetime
import pickle
import pandas as pd
#
# year = now.strftime("%Y")
# print("year:", year)
#
# month = now.strftime("%m")
# print("month:", month)
#
# day = now.strftime("%d")
# print("day:", day)
#
# time = now.strftime("%H:%M:%S")
# print("time:", time)
#

import sys
try:
    import numpy as np
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")


def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


# translation 0 0 0.1
# rotation 0 0 1 3.14



class Mavic (Robot):
    # Constants, empirically found.
    K_VERTICAL_THRUST = 68.5  # with this thrust, the drone lifts.
    # Vertical offset where the robot actually targets to stabilize itself.
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        # P constant of the vertical PID.
    K_ROLL_P = 50.0           # P constant of the roll PID.
    K_PITCH_P = 50.0  #30         # P constant of the pitch PID.

    MAX_YAW_DISTURBANCE = 0.4 #0.4
    MAX_PITCH_DISTURBANCE = -0.5 # -1
    # Precision between the target position and the robot position in meters
    target_precision =  0.5 #0.5

    def __init__(self):
        Robot.__init__(self)

        self.time_step = int(self.getBasicTimeStep())

        # Get and enable devices.
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(0.7)
        motors = [self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor]
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)

        self.current_pose = 6 * [0]  # X, Y, Z, yaw, pitch, roll
        self.target_position = [0, 0, 0]
        self.target_index = 0
        self.target_altitude = 0

        self.angle_left = 0
        self.distance_left = 0
    def set_position(self, pos):
        """
        Set the new absolute position of the robot
        Parameters:
            pos (list): [X,Y,Z,yaw,pitch,roll] current absolute position and angles
        """
        self.current_pose = pos

    def move_to_target(self, waypoints, verbose_movement=False, verbose_target=False):
        """
        Move the robot to the given coordinates
        Parameters:
            waypoints (list): list of X,Y coordinates
            verbose_movement (bool): whether to print remaning angle and distance or not
            verbose_target (bool): whether to print targets or not
        Returns:
            yaw_disturbance (float): yaw disturbance (negative value to go on the right)
            pitch_disturbance (float): pitch disturbance (negative value to go forward)
        """
        target_reached = False
        
        if self.target_position[0:2] == [0, 0]:  # Initialization
            self.target_position[0:2] = waypoints[0]
            if verbose_target:
                print("First target: ", self.target_position[0:2])

        # if the robot is at the position with a precision of target_precision
        if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(self.target_position, self.current_pose[0:2])]):

            self.target_index += 1
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0
            self.target_position[0:2] = waypoints[self.target_index]
            if verbose_target:
                print("Target reached! New target: ",
                      self.target_position[0:2])
            print("Target reached! New target: ", self.target_position[0:2])
            target_reached = True
            # self.target_altitude = 0

        # This will be in ]-pi;pi]
        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        # This is now in ]-2pi;2pi[
        angle_left = self.target_position[2] - self.current_pose[5]
        # Normalize turn angle to ]-pi;pi]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if (angle_left > np.pi):
            angle_left -= 2 * np.pi

        # Turn the robot to the left or to the right according the value and the sign of angle_left
        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        # non proportional and decreasing function
        pitch_disturbance = clamp(
            np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)

        distance_left = np.sqrt(((self.target_position[0] - self.current_pose[0]) ** 2) + (
                (self.target_position[1] - self.current_pose[1]) ** 2))


        if verbose_movement:
            print(f"remaning angle: {angle_left:.2f}, distance: {distance_left:.2f}, "
                  f"X: {self.current_pose[0]:.2f} Y: {self.current_pose[0]:.2f} Z: {self.current_pose[2]:.2f}")

        self.angle_left = angle_left
        self.distance_left = distance_left

        return yaw_disturbance, pitch_disturbance, target_reached

    def __read_sensors__(self):
        # Read sensors
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        x_pos, y_pos, altitude = self.gps.getValues()
        roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
        self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])
        return roll, pitch, yaw, altitude, roll_acceleration, pitch_acceleration

    def __set_velocity__(self, vertical_input, yaw_input, pitch_input, roll_input):
        front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
        front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
        rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
        rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

        self.front_left_motor.setVelocity(front_left_motor_input)
        self.front_right_motor.setVelocity(-front_right_motor_input)
        self.rear_left_motor.setVelocity(-rear_left_motor_input)
        self.rear_right_motor.setVelocity(rear_right_motor_input)

        return front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input

    def run(self):

        print('run')
        t1 = self.getTime()
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m_%d_%H_%M_%S")
        DATA = []

        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0

        # Specify the patrol coordinates
        waypoints = [[-0.77, -1.34]]
        # target altitude of the robot in meters
        self.target_altitude = 1
        target_reached = False

        hovering = False

        while self.step(self.time_step) != -1 and (not target_reached):
            # print(self.time_step)

            roll, pitch, yaw, altitude, roll_acceleration, pitch_acceleration = self.__read_sensors__()

            if altitude > self.target_altitude: # -1
                hovering = True
                # as soon as it reach the target altitude, compute the disturbances to go to the given waypoints.

            if hovering:
                t_check = self.getTime() - t1
                if t_check > 0.01:
                    # print(f'{t_check:.2f}', end = ' ')
                    yaw_disturbance, pitch_disturbance, target_reached = self.move_to_target(
                        waypoints, False, False)
                    t1 = self.getTime()

            clamping_factor = 1
            roll_input = self.K_ROLL_P * clamp(roll, -clamping_factor, clamping_factor) + roll_acceleration + roll_disturbance
            pitch_input = self.K_PITCH_P * clamp(pitch, -clamping_factor, clamping_factor) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance
            clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            motors_inputs = self.__set_velocity__(vertical_input, yaw_input, pitch_input, roll_input)

            DATA.append((self.distance_left, self.angle_left, *self.target_position, *self.current_pose,
                         *motors_inputs))

        print('Landing ')
        front_left, front_right, rear_left, rear_right = motors_inputs
        stop = True
        while self.step(self.time_step) != -1 and stop:
            if self.getTime() - t1 > 2:
                t1 = self.getTime()

                self.front_left_motor.setVelocity(front_left)
                front_left -= 1
                self.front_right_motor.setVelocity(-front_right)
                front_right -= 1
                self.rear_left_motor.setVelocity(-rear_left)
                rear_left -= 1
                self.rear_right_motor.setVelocity(rear_right)
                rear_right -= 1
                stop = front_left > 100 and front_right > 100 and rear_left > 100 and rear_right > 100

        self.front_left_motor.setVelocity(0)
        self.front_right_motor.setVelocity(0)
        self.rear_left_motor.setVelocity(0)
        self.rear_right_motor.setVelocity(0)
        print(';;;;;;;;;;;;;;;')
        with open(f'../../trajectory_{date_time}.pickle', 'wb') as handle:
            pickle.dump(np.array(DATA), handle, protocol=pickle.HIGHEST_PROTOCOL)

        df = pd.DataFrame(np.array(DATA), columns=['dist_left', 'angle_left',
                                                   'target_x', 'target_y', 'target_z',
                                                   'x', 'y', 'z',
                                                   'yaw', 'pitch', 'roll',
                                                   'mot_fl', 'mot_fr', 'mot_rf', 'mot_rr'])

        df.to_csv(f'../../trajectory_{date_time}.csv')
        



# To use this controller, the basicTimeStep should be set to 8 and the defaultDamping
# with a linear and angular damping both of 0.5
robot = Mavic()
robot.run()
