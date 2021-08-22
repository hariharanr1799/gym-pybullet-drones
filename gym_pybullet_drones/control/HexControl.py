import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary
from gym_pybullet_drones.utils.utils import nnlsRPM

class HexPIDControlEul(BaseControl):
    """Generic PID control class without yaw control.

    Based on https://github.com/prfraanje/quadcopter_sim.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.HEXX, DroneModel.HEXP]:
            print("[ERROR] in HexPIDControl.__init__(), HexPIDControl requires DroneModel.HEXP or DroneModel.HEXX, Current Model:", self.DRONE_MODEL)
            exit()
        # self.P_COEFF_FOR = np.array([25, 25, 70])
        # self.I_COEFF_FOR = np.array([0.5, 0.5, 0.5])
        # self.D_COEFF_FOR = np.array([10, 10, 20])
        # self.P_COEFF_TOR = np.array([200, 200, 200])
        # self.I_COEFF_TOR = np.array([0, 0, 0])
        # self.D_COEFF_TOR = np.array([100, 100, 100])

        self.P_COEFF_FOR = np.array([3, 3, 15])
        self.I_COEFF_FOR = np.array([0.05, 0.05, 0.05])
        self.D_COEFF_FOR = np.array([4, 4, 10])
        self.P_COEFF_TOR = np.array([200, 200, 200])
        self.I_COEFF_TOR = np.array([0, 0, 0])
        self.D_COEFF_TOR = np.array([100, 100, 100])

        self.MAX_ROLL_PITCH = np.pi/6
        self.L = self._getURDFParameter('arm')
        self.THRUST2WEIGHT_RATIO = self._getURDFParameter('thrust2weight')
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (6*self.KF))
        self.MAX_THRUST = (6*self.KF*self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (3*self.KM*self.MAX_RPM**2)
        if self.DRONE_MODEL == DroneModel.HEXX:
            self.A = np.array([ [1, 1, 1, 1, 1, 1], [0.5, 1, 0.5, -0.5, -1, -0.5], [-0.866, 0, 0.866, 0.866, 0, -0.866], [-1, 1, -1, 1, -1, 1] ])
        elif self.DRONE_MODEL == DroneModel.HEXP:
            self.A = np.array([ [1, 1, 1, 1, 1, 1], [0, 0.866, 0.866, 0, -0.866, -0.866], [-1, -0.5, 0.5, 1, 0.5, -0.5], [-1, 1, -1, 1, -1, 1] ])
        self.INV_A = np.linalg.pinv(self.A)
        self.B_COEFF = np.array([1/self.KF, 1/(self.KF*self.L), 1/(self.KF*self.L), 1/self.KM])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
    
    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_hexPIDPositionControl()` and `_hexPIDAttitudeControl()`.
        Parameters `cur_ang_vel`, `target_rpy`, `target_vel`, and `target_rpy_rates` are unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        if target_rpy[2]!=0:
            print("\n[WARNING] ctrl it", self.control_counter, "in HexPIDControl.computeControl(), desired yaw={:.0f}deg but locked to 0. for DroneModel.HEX".format(target_rpy[2]*(180/np.pi)))
        thrust, computed_target_rpy, pos_e = self._hexPIDPositionControl(control_timestep,
                                                                            cur_pos,
                                                                            cur_quat,
                                                                            target_pos
                                                                            )
        rpm = self._hexPIDAttitudeControl(control_timestep,
                                             thrust,
                                             cur_quat,
                                             computed_target_rpy
                                             )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    ################################################################################

    def _hexPIDPositionControl(self,
                                  control_timestep,
                                  cur_pos,
                                  cur_quat,
                                  target_pos
                                  ):
        """Simple PID position control (with yaw fixed to 0).

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        pos_e = target_pos - np.array(cur_pos).reshape(3)
        d_pos_e = (pos_e - self.last_pos_e) / control_timestep
        self.last_pos_e = pos_e
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        #### PID target thrust #####################################
        target_force = self.m*(np.array([0, 0, self.G]) \
                       + np.multiply(self.P_COEFF_FOR, pos_e) \
                       + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                       + np.multiply(self.D_COEFF_FOR, d_pos_e))
        target_rpy = np.zeros(3)
        sign_z =  np.sign(target_force[2])
        if sign_z == 0:
            sign_z = 1
        #### Target rotation #######################################
        target_rpy[0] = np.arcsin(-sign_z*target_force[1] / np.linalg.norm(target_force))
        target_rpy[1] = np.arctan2(sign_z*target_force[0], sign_z*target_force[2])
        target_rpy[2] = 0.
        target_rpy[0] = np.clip(target_rpy[0], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
        target_rpy[1] = np.clip(target_rpy[1], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        thrust = np.clip(np.dot(cur_rotation, target_force), -self.MAX_THRUST, self.MAX_THRUST)
        return thrust[2], target_rpy, pos_e

    ################################################################################

    def _hexPIDAttitudeControl(self,
                                  control_timestep,
                                  thrust,
                                  cur_quat,
                                  target_rpy
                                  ):
        """Simple PID attitude control (with yaw fixed to 0).

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the computed the target roll, pitch, and yaw.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        rpy_e = target_rpy - np.array(cur_rpy).reshape(3,)
        if rpy_e[2] > np.pi:
            rpy_e[2] = rpy_e[2] - 2*np.pi
        if rpy_e[2] < -np.pi:
            rpy_e[2] = rpy_e[2] + 2*np.pi
        d_rpy_e = (rpy_e - self.last_rpy_e) / control_timestep
        self.last_rpy_e = rpy_e
        self.integral_rpy_e = self.integral_rpy_e + rpy_e*control_timestep
        #### PID target torques ####################################
        target_torques = np.multiply(np.array([self.ixx, self.iyy, self.izz]), np.multiply(self.P_COEFF_TOR, rpy_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e) \
                         + np.multiply(self.D_COEFF_TOR, d_rpy_e))
        return nnlsRPM(thrust=thrust,
                       x_torque=target_torques[0],
                       y_torque=target_torques[1],
                       z_torque=target_torques[2],
                       counter=self.control_counter,
                       max_thrust=self.MAX_THRUST,
                       max_xy_torque=self.MAX_XY_TORQUE,
                       max_z_torque=self.MAX_Z_TORQUE,
                       a=self.A,
                       inv_a=self.INV_A,
                       b_coeff=self.B_COEFF,
                       gui=True
                       )
 
class HexPIDControlQuat(BaseControl):
    """
    PID Controller Applied to Hexacopter Flight
    Alaimo, A., Artale, V., Milazzo, C. L. R., & Ricciardello, A. (2013). PID Controller Applied to Hexacopter Flight. Journal of Intelligent & Robotic Systems, 73(1-4), 261–270. doi:10.1007/s10846-013-9947-y 
    
    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.HEXP:
            print("[ERROR] in HexPIDControl.__init__(), HexPIDControl requires DroneModel.HEX")
            exit()
        self.P_COEFF_FOR = np.array([.2, 1, 10])
        self.I_COEFF_FOR = np.array([.02, .13, 5])
        self.D_COEFF_FOR = np.array([.75, 3.1, 10])
        self.P_COEFF_TOR = np.array([.25, .25, .3])
        self.I_COEFF_TOR = np.array([.0001, .0001, .0001])
        self.D_COEFF_TOR = np.array([-0.25, -0.25, -0.25])
        self.MAX_ROLL_PITCH = np.pi/6
        self.L = self._getURDFParameter('arm')
        self.THRUST2WEIGHT_RATIO = self._getURDFParameter('thrust2weight')
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (6*self.KF))
        self.MAX_THRUST = (6*self.KF*self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (3*self.KM*self.MAX_RPM**2)
        self.MIXER_MATRIX = np.array([[1/(6*self.KF), 0, -2/(5*self.KF*self.L), -1/(10*self.KM)], \
                                      [1/(6*self.KF), 1/(3*self.KF*self.L), -1/(5*self.KF*self.L), 1/(5*self.KM)], \
                                      [1/(6*self.KF), 1/(3*self.KF*self.L), 1/(5*self.KF*self.L), -1/(5*self.KM)], \
                                      [1/(6*self.KF), 0, 2/(5*self.KF*self.L), 1/(10*self.KM)], \
                                      [1/(6*self.KF), -1/(3*self.KF*self.L), 1/(5*self.KF*self.L), -1/(5*self.KM)], \
                                      [1/(6*self.KF), -1/(3*self.KF*self.L), -1/(5*self.KF*self.L), 1/(5*self.KM)]])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
    
    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_hexPIDPositionControl()` and `_hexPIDAttitudeControl()`.
        Parameters `cur_ang_vel`, `target_rpy`, `target_vel`, and `target_rpy_rates` are unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        
        thrust, computed_target_quat, pos_e = self._hexPIDPositionControl(control_timestep,
                                                                            cur_pos,
                                                                            cur_quat,
                                                                            target_pos,
                                                                            )
        rpm = self._hexPIDAttitudeControl(control_timestep,
                                             thrust,
                                             cur_quat,
                                             computed_target_quat
                                             )
        
        return rpm, pos_e, computed_target_quat[3] - cur_quat[3]

    ################################################################################

    def _hexPIDPositionControl(self,
                                  control_timestep,
                                  cur_pos,
                                  cur_quat,
                                  target_pos,
                                  ):
        """Simple PID position control (with yaw fixed to 0).

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        pos_e = target_pos - np.array(cur_pos).reshape(3)
        d_pos_e = (pos_e - self.last_pos_e) / control_timestep
        self.last_pos_e = pos_e
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep

        dx, dy, dz = np.multiply(self.P_COEFF_FOR, pos_e) \
                       + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                       + np.multiply(self.D_COEFF_FOR, d_pos_e)
        
        target_quat = np.zeros(4)
        target_quat[3] = cur_quat[3]

        target_quat[0] = max(0,np.sqrt(0.5 + (dz+self.G)/(2*np.sqrt(dx**2 + dy**2 + (dz+self.G)**2)) - cur_quat[3]**2))
        thrust = self.m*(dz + self.G)/(2*(cur_quat[0]**2+cur_quat[3]**2)-1)
        target_quat[1] = self.m*(dx*cur_quat[3] - dy*cur_quat[0])/(2*(cur_quat[0]**2+cur_quat[3]**2)*thrust)
        target_quat[2] = self.m*(dx*cur_quat[0] + dy*cur_quat[3])/(2*(cur_quat[0]**2+cur_quat[3]**2)*thrust)
        
        return thrust, target_quat, pos_e

    ################################################################################
    
    def _hexPIDAttitudeControl(self,
                                  control_timestep,
                                  thrust,
                                  cur_quat,
                                  target_quat
                                  ):
        """Simple PID attitude control (with yaw fixed to 0).

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the computed the target roll, pitch, and yaw.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        quat_mat = np.array([[-target_quat[0], -target_quat[3], target_quat[2], target_quat[1]], \
                             [target_quat[3], -target_quat[0], -target_quat[1], target_quat[2]], \
                             [-target_quat[2], target_quat[1], -target_quat[0], target_quat[3]], \
                             [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]])
        
        qe1, qe2, qe3, qe4 = np.dot(quat_mat, cur_quat)

        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        rpy_e = p.getEulerFromQuaternion(target_quat) - np.array(cur_rpy).reshape(3,)
        if rpy_e[2] > np.pi:
            rpy_e[2] = rpy_e[2] - 2*np.pi
        if rpy_e[2] < -np.pi:
            rpy_e[2] = rpy_e[2] + 2*np.pi
        rollrate, pitchrate, yawrate = (rpy_e - self.last_rpy_e) / control_timestep
        
        target_torques = np.zeros(3)
        target_torques[0] = self.ixx * (self.P_COEFF_TOR[0]*qe1*qe4 + self.D_COEFF_TOR[0]*rollrate)
        target_torques[1] = self.iyy * (2*self.P_COEFF_TOR[1]*qe2*qe4 + self.D_COEFF_TOR[1]*pitchrate)
        target_torques[2] = self.izz * (2*self.P_COEFF_TOR[2]*qe3*qe4 + self.D_COEFF_TOR[2]*yawrate)

        sq_rpm = np.dot(self.MIXER_MATRIX, np.array([thrust, target_torques[0], target_torques[1], target_torques[2]]))
        return np.sqrt(sq_rpm)
