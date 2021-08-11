"""Script demonstrating the ground effect contribution.

The simulation is run by a `CtrlAviary` environment.

Example
-------
In a terminal, run as:

    $ python groundeffect.py

Notes
-----
The drone altitude tracks a sinusoid, near the ground plane.

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.HexControl import *
from gym_pybullet_drones.control.ModelPredControl import HexMPC
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


def plotContactData():
    plt.figure()
    plt.subplot(221)
    plt.title('Normal Force (N)')
    plt.plot(contact_forces)
    plt.subplot(222)
    plt.title('Contact Distance (m)')
    plt.plot(contact_distance)
    plt.subplot(223)
    plt.title('Friction_x (N)')
    plt.plot(friction_x)
    plt.subplot(224)
    plt.title('Friction_y (N)')
    plt.plot(friction_y)
    plt.show()    


if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Ground effect script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--gui',                default=True,               type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,              type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,               type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,              type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,              type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,              type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=250,                type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=250,                 type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--sensor_freq_hz',     default=250,                 type=int,           help='Sensor frequency in Hz (default: 30)', metavar='')
    parser.add_argument('--duration_sec',       default=5,                  type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--visualize_box',      default=True,               type=str2bool,      help='Visualize the boxes (default: True)', metavar='')
    parser.add_argument('--drone_model',        default=DroneModel.HEXP,    type=DroneModel,    help='Drone Model (default: True)', metavar='')
    ARGS = parser.parse_args()

    #### Box parameters ########################################
    BOX_SIDE = 0.2 # m
    TIME_SIDE = 5 #s
    INIT_XYZ = np.array([0,0,0.2]).reshape(1,3)
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Create the environment ################################
    env = CtrlAviary(drone_model=ARGS.drone_model,
                     num_drones=1,
                     num_rotors=6,
                     rotor_angle=0, #degrees
                     initial_xyzs=INIT_XYZ,
                     physics=Physics.PYB,
                     neighbourhood_radius=10,
                     freq=ARGS.simulation_freq_hz,
                     aggregate_phy_steps=AGGR_PHY_STEPS,
                     gui=ARGS.gui,
                     record=ARGS.record_video,
                     obstacles=ARGS.obstacles,
                     user_debug_gui=ARGS.user_debug_gui
                     )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=1
                    )

    # time.sleep(10)

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    SENSOR_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.sensor_freq_hz))
    action = {"0": np.array([0]*6)}
    START = time.time()
    ctrl_counter = 0
    line_counter = 0
    corner_ind = 0
    
    uav_pos = INIT_XYZ.reshape(3,)
    TARGET_POS = INIT_XYZ.reshape(3,)

    #### Initialize the controller #############################
    ctrl = HexMPC(drone_model=ARGS.drone_model, init_xyz=INIT_XYZ)
    
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        
        state = obs["0"]["state"]
        
        TARGET_POS = [0,0,1]

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            action["0"] = ctrl.computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                             state=state,
                                                             target_pos=TARGET_POS,
                                                             )
            print("====>", i)

            #### Go to the next way point and loop #####################
            ctrl_counter = ctrl_counter + 1 #if ctrl_counter < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state= obs["0"]["state"],
                   control=np.hstack([TARGET_POS, np.zeros(9)])
                   )

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    # logger.save_as_csv("gnd") # Optional CSV save

    #### Print Contact Details #################################
    contact_forces = []
    contact_distance = []
    friction_x = []
    friction_y = []
    for d in env._getContactData():
        if len(d) == 0:
            contact_forces.append(0)
            contact_distance.append(1)
            friction_x.append(0)
            friction_y.append(0)
        else:
            contact_forces.append(d[9])
            contact_distance.append(d[8])
            friction_x.append(d[12])
            friction_y.append(d[10])

    #### Plot the simulation results ###########################
    if ARGS.plot:
        # plotContactData()
        logger.plot()
