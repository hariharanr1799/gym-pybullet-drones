import numpy as np
from casadi import *
from numpy.core.numeric import full


def Mnorm(P, Q):
    return np.dot(np.dot(np.transpose(P),Q),P)


SIM_TIME = 70           # Max simulation time
T = 0.1                 # Sampling time
N = 50                  # Prediction Horizon
ROB_DIAM = 10           # Robot Diameter
CTRL_H = 1              # Control Horizon

N_STATES = 8            # Number of state variables
N_CONTROLS = 4          # Number of control inputs

if CTRL_H > N:
    print("Control Horizon cannot be greater than prediction horizon!")
    exit()

x0 = np.array([-4.5,15,35,0,5,0,0,0])

ref = []
for t_ in range(int(SIM_TIME//T)):
    t = t_*T
    ref.append([x0[0]+5*t,15,35,0,5,0,0,0])

Q = np.diag([5, 5, 5, 1, 0, 0, 0, 0])   # Weight matrix of states
R = np.diag([1, 1, 1, 0.1])             # Weight matrix of controls

CIRCULAR_VARS = [3]

#### SET STATE AND CONTROL INPUT BOUNDS ####
STATE_BOUNDS = np.array([
                            [-inf, inf],
                            [-inf, inf],
                            [0, 100],
                            [-inf, inf],
                            [-3, 3],
                            [-3, 3],
                            [-3, 3],
                            [-pi, pi]
                        ])

CONTROL_BOUNDS = np.array([
                            [-5, 5],
                            [-5, 5],
                            [-5, 5],
                            [-5, 5]
                        ])

states = SX.sym('states', N_STATES)       # States
controls = SX.sym('controls', N_CONTROLS)     # Controls

#### Dynamic Model ####
xdot = np.array([
                    states[4]*cos(states[3]) - states[5]*sin(states[3]),
                    states[4]*sin(states[3]) + states[5]*cos(states[3]),
                    states[6],
                    states[7],
                    (-states[4] + controls[0])/0.8355,
                    (-states[5] + controls[1])/0.7701,
                    (-states[6] + controls[2])/0.5013,
                    (-states[7] + controls[3])/0.5142,
                ])

f = Function('f',[states, controls], xdot)  # Nonlinear mapping function f(x,u) representing the dynamics

U = SX.sym('U', N_CONTROLS, N)  # Decision control variables
P = SX.sym('P', 1+N_STATES*2)   # Parameters (Time, initial state of the robot, and the reference state)
X = SX.sym('X', N_STATES, N+1)  # Vector representing the states over the optimization problem.

obj = 0

st = X[:,0]
g = [st - P[:N_STATES]]         # Initial condition constraint

for k in range(N):
    st = X[:,k]
    con = U[:,k]
    err = st - P[N_STATES:N_STATES*2-1]
    for i in CIRCULAR_VARS:
        err[i] = (err[i]+pi)%(2*pi) - pi
    
    obj += (Mnorm(err, Q) + Mnorm(con, R))
    st_next = X[:,k+1]
    f_value = f(st,con)
    st_next_euler = st + (T*f_value)
    for i in CIRCULAR_VARS:
        st_next_euler[i] = (st_next_euler[i]+pi)%(2*pi) - pi
    
    g.append(st_next_euler-st_next)

lbg = [-inf,-inf,-inf,-inf,-inf,-inf]
ubg = [0,0,0,0,0,0]

OPT_vars = [reshape(X,N_STATES*(N+1),1),reshape(U,N_CONTROLS*N,1)]

nlp = {
    'x': OPT_vars,
    'f': obj,
    'g': g,
    'p': P
    }

opts = {'ipopt': {'print_level': 0, 'max_iter': 200, 'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6}}
solver = nlpsol('solver','ipopt',nlp,opts)

args = {'lbg': np.zeros(N_STATES*(N+1)-1), 'ubg': np.zeros(N_STATES*(N+1)-1), 'lbx': np.zeros(N_STATES*(N+1)+N_CONTROLS*N), 'ubx': np.zeros(N_STATES*(N+1)+N_CONTROLS*N)}

for i in range(N_STATES):
    for j in range(i,N_STATES*(N+1),N_STATES):
        args['lbx'][j] = STATE_BOUNDS[i][0]
        args['ubx'][j] = STATE_BOUNDS[i][1]

for i in range(N_CONTROLS):
    for j in range(N_STATES(N+1)-1+i,N_CONTROLS*N,N_CONTROLS):
        args['lbx'][j] = CONTROL_BOUNDS[i][0]
        args['ubx'][j] = CONTROL_BOUNDS[i][1]


mpciter = 0
xx1 = []
u_cl = []
t0 = 0
xx = [x0]
t[0] = t0

u0 = np.zeros(N,N_CONTROLS)
X0 = np.transpose(repmat(x0,1,N+1))

xsar = []

while (mpciter < SIM_TIME/T):
    xs = ref[mpciter+1]
    args['p'] = [x0, xs, t0]
    args['x0'] = [reshape(np.transpose(X0), N_STATES*(N+1),1), reshape(np.transpose(u0), N_CONTROLS*N, 1)]
    sol = solver(x0 = args['x0'], lbx = args['lbx'], ubx = args['ubx'], lbg = args['lbg'], ubg = args['ubg'], p = args['p'])

    u = np.transpose(reshape(np.transpose(full(sol['x'][N_STATES*(N+1):])),N_CONTROLS,N))

    for i in range(1,CTRL_H):
        
        mpciter += 1
