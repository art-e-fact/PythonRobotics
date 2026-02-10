import numpy as np
import math
import matplotlib.pyplot as plt


## Estimation parameters of EKF
Q = np.diag([1.0, 1.0])**2 # observation x, y position (GPS) covariance
R = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0])**2 # predict state covariance

## Simulation parameters
Qsim = np.diag([0.5, 0.5])**2 # observation covariance
Rsim = np.diag([1.0, np.deg2rad(30.0)])**2 # input covariance

DT = 0.1 # time tick (s)
SIM_TIME = 50.0 # simulation duration (s)


def plot_covariance_ellipse(xEst, PEst):
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2*math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a*math.cos(it) for it in t]
    y = [b*math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    R = np.array([
        [math.cos(angle), math.sin(angle)],
        [-math.sin(angle), math.cos(angle)]
    ])
    fx = R.dot(np.array([[x, y]]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")

def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]
                  ])
    B = np.array([[DT * math.cos(x[2,0]), 0],
                  [DT * math.sin(x[2,0]), 0],
                  [0.0, DT],
                  [1.0, 0.0],
                  ])
    x = F.dot(x) + B.dot(u)
    return x

def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H.dot(x)
    
    return z

def jacobF(x, u):
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1, 0, -v*math.sin(yaw)*DT, math.cos(yaw)*DT],
        [0, 1, v*math.cos(yaw)*DT, math.sin(yaw)*DT],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return jF

def jacobH(x):
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    return jH

def ekf_estimation(xEst, PEst, z, u):
    # predict
    xPred = motion_model(xEst, u)    
    jF = jacobF(xPred, u)
    PPred = jF.dot(PEst).dot(jF.T) + R

    # update
    jH = jacobH(xPred)
    zPred = observation_model(xPred)
    y = z.T - zPred
    S = jH.dot(PPred).dot(jH.T) + Q
    K = PPred.dot(jH.T).dot(np.linalg.inv(S))
    xEst = xPred + K.dot(y)
    PEst = (np.eye(len(xEst)) - K.dot(jH).dot(PPred))
    
    return xEst, PEst

def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x,y
    zx = xTrue[0,0] + np.random.randn() * Qsim[0, 0]
    zy = xTrue[1,0] + np.random.randn() * Qsim[1, 1]
    z = np.array([[zx, zy]])

    # add noise to input
    ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0]
    ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1]
    ud = np.array([[ud1, ud2]]).T

    xd = motion_model(xd, ud)
    
    return xTrue, z, xd, ud 

def calc_input():
    v = 1.0 # (m/s)
    v_yaw = 0.1 # (rad/s)
    u = np.array([[v, v_yaw]]).T
    return u

def main():
    print('[EKF] Start.')
    time = 0.0

    ## initial params     
    xEst = np.zeros((4,1)) # estimated state vector (x, y, yaw, v)
    xTrue = np.zeros((4,1)) # ground truth state vector
    xDR = np.zeros((4,1)) # dead reckoning (using only noisy input)
    PEst = np.eye(4) # state covariance matrix

    ## history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((1, 2))

    while time <= SIM_TIME:
        time += DT
        
        # using constant input
        u = calc_input()
        
        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        ## store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.vstack((hz, z))

        ## show animations
        plt.cla()
        plt.plot(hz[:, 0], hz[:, 1], ".g")
        plt.plot(hxTrue[0, :].flatten(),
                 hxTrue[1, :].flatten(), "-b")
        plt.plot(hxDR[0, :].flatten(),
                 hxDR[1, :].flatten(), "-k")
        plt.plot(hxEst[0, :].flatten(),
                 hxEst[1, :].flatten(), "-r")
        plot_covariance_ellipse(xEst, PEst)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)

if __name__ == '__main__':
    main()
