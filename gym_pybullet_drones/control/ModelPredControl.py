import math
import numpy as np
import pybullet as p
import do_mpc
from casadi import *
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary
from gym_pybullet_drones.utils.utils import nnlsRPM

class HexMPC_(BaseControl):

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8,
                 init_xyz=np.zeros((12,1)),
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

        self.N_HORIZON = 10
        self.TIME_STEP = 0.1
        self.Q = np.array([1,1,1,1,1,1,0,0,0,0,0,0])*2
        self.ref = np.array([0,0,1,0,0,0,0,0,0,0,0,0])
        
        self.MPC_init(init_xyz)
        self.reset()

    ################################################################################

    def updateSetpoint(self, t_now):
        for k in range(self.N_HORIZON+1):
            self.sp_template['_tvp',k,'xd'] = np.array([0.1,0,1,0,0,0,0,0,0,0,0,0])

        return self.sp_template

    def MPC_init(self, init_xyz):
        self.model = do_mpc.model.Model('continuous')

        self.x = self.model.set_variable(var_type='_x', var_name='x', shape=(12,1))
        self.u = self.model.set_variable(var_type='_u', var_name='u', shape=(4,1))
        self.xd = self.model.set_variable(var_type='_tvp', var_name='xd', shape=(12,1))

        # self.dx = vertcat(
        #     self.x[6]*np.cos(self.x[4])*np.cos(self.x[5]) + self.x[7]*(np.sin(self.x[3])*np.sin(self.x[4])*np.cos(self.x[5])-np.cos(self.x[3])*np.sin(self.x[5])) + self.x[8]*(np.cos(self.x[3])*np.sin(self.x[4])*np.cos(self.x[5]) + np.sin(self.x[3])*np.sin(self.x[5])),
        #     self.x[6]*np.cos(self.x[4])*np.sin(self.x[5]) + self.x[7]*(np.sin(self.x[3])*np.sin(self.x[4])*np.sin(self.x[5])+np.cos(self.x[3])*np.cos(self.x[5])) + self.x[8]*(np.cos(self.x[3])*np.sin(self.x[4])*np.sin(self.x[5]) - np.sin(self.x[3])*np.cos(self.x[5])),
        #     -self.x[6]*np.sin(self.x[4]) + self.x[7]*np.sin(self.x[3])*np.cos(self.x[4]) + self.x[8]*np.cos(self.x[3])*np.cos(self.x[4]),
        #     self.x[9] + self.x[10]*np.sin(self.x[3])*tan(self.x[4]) + self.x[11]*np.cos(self.x[3])*tan(self.x[4]),
        #     self.x[10]*np.cos(self.x[3]) - self.x[11]*np.sin(self.x[3]),
        #     self.x[10]*np.sin(self.x[3])/np.cos(self.x[4]) + self.x[11]*np.cos(self.x[3])/np.cos(self.x[4]),
        #     self.x[11]*self.x[7] - self.x[10]*self.x[8] + self.G*sin(self.x[5]),
        #     self.x[9]*self.x[8] - self.x[11]*self.x[6] - self.G*cos(self.x[5])*sin(self.x[4]),
        #     self.x[10]*self.x[6] - self.x[9]*self.x[7] - self.G*cos(self.x[5])*cos(self.x[4]) + self.u[0]/self.m,
        #     ((self.iyy-self.izz)*self.x[10]*self.x[11] + self.u[1])/self.ixx,
        #     ((self.izz-self.ixx)*self.x[9]*self.x[11] + self.u[2])/self.iyy,
        #     ((self.ixx-self.iyy)*self.x[9]*self.x[10] + self.u[3])/self.izz
        # )

        self.dx = vertcat(
            self.x[6],
            self.x[7],
            self.x[8],
            self.x[9],
            self.x[10],
            self.x[11],
            self.G*self.x[4],
            -self.G*self.x[3],
            self.u[0]/self.m - self.G,
            self.u[1]/self.ixx,
            self.u[2]/self.iyy,
            self.u[3]/self.izz
        )

        self.model.set_rhs('x', self.dx)

        self.model.setup()

        self.mpc = do_mpc.controller.MPC(self.model)

        setup_mpc = {
            'n_horizon': self.N_HORIZON,
            't_step': self.TIME_STEP,
            'n_robust': 0,
            'store_full_solution': True,
        }

        self.mpc.set_param(**setup_mpc)

        # surpress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
        # self.mpc.set_param(nlpsol_opts = surpress_ipopt)

        mterm = sum1(self.Q*(self.x-self.xd)**2)
        lterm = mterm

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(
            u=np.array([1e-2,1e-2,1e-2,1e-2])
        )

        # Lower bounds on states:
        self.mpc.bounds['lower','_x', 'x'] = np.array([-inf,-inf,0,-np.pi/3,-np.pi/2,-np.pi,-5,-5,-5,-np.pi/4,-np.pi/4,-np.pi/4])
        # Upper bounds on states
        self.mpc.bounds['upper','_x', 'x'] = np.array([inf,inf,100,np.pi/3,np.pi/2,np.pi,5,5,5,np.pi/4,np.pi/4,np.pi/4])
        # Lower bounds on inputs:
        self.mpc.bounds['lower','_u', 'u'] = np.array([0,-self.MAX_XY_TORQUE,-self.MAX_XY_TORQUE,-self.MAX_Z_TORQUE])
        # Lower bounds on inputs:
        self.mpc.bounds['upper','_u', 'u'] = np.array([self.MAX_THRUST,self.MAX_XY_TORQUE,self.MAX_XY_TORQUE,self.MAX_Z_TORQUE])
        
        self.sp_template = self.mpc.get_tvp_template()
        
        self.mpc.set_tvp_fun(self.updateSetpoint)

        self.mpc.setup()
        x0 = np.zeros((12,1))
        x0[:3] = np.transpose(init_xyz[:3])
        self.mpc.x0 = x0.reshape(-1,1)

        self.mpc.set_initial_guess()


    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        try:
            self.mpc.reset_history()
        except:
            pass
    
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
        self.ref[:3] = target_pos

        cur_rpy = list(p.getEulerFromQuaternion(cur_quat))

        x = np.hstack((cur_pos, cur_rpy, cur_vel, cur_ang_vel)).reshape(-1,1)
        u = self.mpc.make_step(x)

        return nnlsRPM(thrust=u[0][0],
                       x_torque=u[1][0],
                       y_torque=u[2][0],
                       z_torque=u[3][0],
                       counter=self.control_counter,
                       max_thrust=self.MAX_THRUST,
                       max_xy_torque=self.MAX_XY_TORQUE,
                       max_z_torque=self.MAX_Z_TORQUE,
                       a=self.A,
                       inv_a=self.INV_A,
                       b_coeff=self.B_COEFF,
                       gui=True
                       )

class HexMPC(BaseControl):

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8,
                 init_xyz=np.zeros((12,1)),
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
        
        self.MAX_ROLL_PITCH = np.pi/4
        self.L = self._getURDFParameter('arm')
        self.THRUST2WEIGHT_RATIO = self._getURDFParameter('thrust2weight')
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (6*self.KF))
        self.MAX_THRUST = (6*self.KF*self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)

        self.N_HORIZON = 10
        self.TIME_STEP = 0.1
        self.Q = np.array([1,1,1,1,1,1,0,0,0,0,0,0])*2
        self.ref = np.array([0,0,1,0,0,0,0,0,0,0,0,0])
        
        self.MPC_init(init_xyz)
        self.reset()

    ################################################################################

    def updateSetpoint(self, t_now):
        for k in range(self.N_HORIZON+1):
            self.sp_template['_tvp',k,'xd'] = np.array([0.1,0,1,0,0,0,0,0,0,0,0,0])

        return self.sp_template

    def MPC_init(self, init_xyz):
        self.model = do_mpc.model.Model('continuous')

        self.x = self.model.set_variable(var_type='_x', var_name='x', shape=(12,1))
        self.u = self.model.set_variable(var_type='_u', var_name='u', shape=(6,1))
        self.xd = self.model.set_variable(var_type='_tvp', var_name='xd', shape=(12,1))

        # self.dx = vertcat(
        #     self.x[6]*np.cos(self.x[4])*np.cos(self.x[5]) + self.x[7]*(np.sin(self.x[3])*np.sin(self.x[4])*np.cos(self.x[5])-np.cos(self.x[3])*np.sin(self.x[5])) + self.x[8]*(np.cos(self.x[3])*np.sin(self.x[4])*np.cos(self.x[5]) + np.sin(self.x[3])*np.sin(self.x[5])),
        #     self.x[6]*np.cos(self.x[4])*np.sin(self.x[5]) + self.x[7]*(np.sin(self.x[3])*np.sin(self.x[4])*np.sin(self.x[5])+np.cos(self.x[3])*np.cos(self.x[5])) + self.x[8]*(np.cos(self.x[3])*np.sin(self.x[4])*np.sin(self.x[5]) - np.sin(self.x[3])*np.cos(self.x[5])),
        #     -self.x[6]*np.sin(self.x[4]) + self.x[7]*np.sin(self.x[3])*np.cos(self.x[4]) + self.x[8]*np.cos(self.x[3])*np.cos(self.x[4]),
        #     self.x[9] + self.x[10]*np.sin(self.x[3])*tan(self.x[4]) + self.x[11]*np.cos(self.x[3])*tan(self.x[4]),
        #     self.x[10]*np.cos(self.x[3]) - self.x[11]*np.sin(self.x[3]),
        #     self.x[10]*np.sin(self.x[3])/np.cos(self.x[4]) + self.x[11]*np.cos(self.x[3])/np.cos(self.x[4]),
        #     self.x[11]*self.x[7] - self.x[10]*self.x[8] + self.G*sin(self.x[5]),
        #     self.x[9]*self.x[8] - self.x[11]*self.x[6] - self.G*cos(self.x[5])*sin(self.x[4]),
        #     self.x[10]*self.x[6] - self.x[9]*self.x[7] - self.G*cos(self.x[5])*cos(self.x[4]) + self.u[0]/self.m,
        #     ((self.iyy-self.izz)*self.x[10]*self.x[11] + self.u[1])/self.ixx,
        #     ((self.izz-self.ixx)*self.x[9]*self.x[11] + self.u[2])/self.iyy,
        #     ((self.ixx-self.iyy)*self.x[9]*self.x[10] + self.u[3])/self.izz
        # )

        self.dx = vertcat(
            self.x[6],
            self.x[7],
            self.x[8],
            self.x[9],
            self.x[10],
            self.x[11],
            self.G*self.x[4],
            -self.G*self.x[3],
            self.u[0]/self.m - self.G,
            self.u[1]/self.ixx,
            self.u[2]/self.iyy,
            self.u[3]/self.izz
        )

        self.model.set_rhs('x', self.dx)

        self.model.setup()

        self.mpc = do_mpc.controller.MPC(self.model)

        setup_mpc = {
            'n_horizon': self.N_HORIZON,
            't_step': self.TIME_STEP,
            'n_robust': 0,
            'store_full_solution': True,
        }

        self.mpc.set_param(**setup_mpc)

        # surpress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
        # self.mpc.set_param(nlpsol_opts = surpress_ipopt)

        mterm = sum1(self.Q*(self.x-self.xd)**2)
        lterm = mterm

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(
            u=np.array([1e-2,1e-2,1e-2,1e-2])
        )

        # Lower bounds on states:
        self.mpc.bounds['lower','_x', 'x'] = np.array([-inf,-inf,0,-np.pi/3,-np.pi/2,-np.pi,-5,-5,-5,-np.pi/4,-np.pi/4,-np.pi/4])
        # Upper bounds on states
        self.mpc.bounds['upper','_x', 'x'] = np.array([inf,inf,100,np.pi/3,np.pi/2,np.pi,5,5,5,np.pi/4,np.pi/4,np.pi/4])
        # Lower bounds on inputs:
        self.mpc.bounds['lower','_u', 'u'] = np.array([0,-self.MAX_XY_TORQUE,-self.MAX_XY_TORQUE,-self.MAX_Z_TORQUE])
        # Lower bounds on inputs:
        self.mpc.bounds['upper','_u', 'u'] = np.array([self.MAX_THRUST,self.MAX_XY_TORQUE,self.MAX_XY_TORQUE,self.MAX_Z_TORQUE])
        
        self.sp_template = self.mpc.get_tvp_template()
        
        self.mpc.set_tvp_fun(self.updateSetpoint)

        self.mpc.setup()
        x0 = np.zeros((12,1))
        x0[:3] = np.transpose(init_xyz[:3])
        self.mpc.x0 = x0.reshape(-1,1)

        self.mpc.set_initial_guess()


    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        try:
            self.mpc.reset_history()
        except:
            pass
    
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
        self.ref[:3] = target_pos

        cur_rpy = list(p.getEulerFromQuaternion(cur_quat))

        x = np.hstack((cur_pos, cur_rpy, cur_vel, cur_ang_vel)).reshape(-1,1)
        u = self.mpc.make_step(x)

        return nnlsRPM(thrust=u[0][0],
                       x_torque=u[1][0],
                       y_torque=u[2][0],
                       z_torque=u[3][0],
                       counter=self.control_counter,
                       max_thrust=self.MAX_THRUST,
                       max_xy_torque=self.MAX_XY_TORQUE,
                       max_z_torque=self.MAX_Z_TORQUE,
                       a=self.A,
                       inv_a=self.INV_A,
                       b_coeff=self.B_COEFF,
                       gui=True
                       )