import pytest
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
module_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../localization/extended_kalman_filter/")
sys.path.append(module_path)
import extended_kalman_filter as EKF


@pytest.fixture(scope='module')
def init_fixture():
    xEst = np.zeros((4, 1)) # estimated state vector (x, y, yaw, v)
    xTrue = np.zeros((4,1)) # ground truth state vector
    xDR = np.zeros((4,1)) # dead reckoning using only noisy input + known motion model
    PEst = np.eye(4) # state covariance matrix

    ## history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((1,2))
    
    yield xEst, xTrue, xDR, PEst, hxEst, hxTrue, hxDR, hz

def calc_dist(xEst, xTrue):
    return np.sqrt((xEst[0]-xTrue[0])**2 + (xEst[1]-xTrue[1])**2)[0]


## estimation params of EKF
# observation covariance (x, y GPS position)
@pytest.mark.parametrize("Q", [
    np.diag([1.0, 1.0])**2
])
# predicted state / process covariance
@pytest.mark.parametrize("R", [
    np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0])**2
])

## simulation params
# observation noise
@pytest.mark.parametrize("Qsim", [
    np.diag([0.2, 0.2])**2,
    np.diag([0.5, 0.5])**2,
    np.diag([0.8, 0.8])**2,
])
# input noise
@pytest.mark.parametrize("Rsim", [
    np.diag([1.0, np.deg2rad(30.0)])**2
])
# time tick
@pytest.mark.parametrize("dt", [0.1])
# total simulation time
@pytest.mark.parametrize("SIM_TIME", [30.0])

def test_EKF(init_fixture, Q, R, Qsim, Rsim, dt, SIM_TIME):
    print('[EKF] Start.')

    xEst, xTrue, xDR, PEst, hxEst, hxTrue, hxDR, hz = init_fixture
    
    time = 0.0    
    while time <= SIM_TIME:
        time += dt

        # using constant input
        u = EKF.calc_input()
        
        xTrue, z, xDR, ud = EKF.observation(xTrue, xDR, u)
        
        xEst, PEst = EKF.ekf_estimation(xEst, PEst, z, ud)

        est_error = calc_dist(xEst, xTrue)
        DR_error = calc_dist(xDR, xTrue)

        if round(time,1).is_integer():
            print(f"[t={time:.2f}] EKF distance = {est_error:.2f} - DR distance = {DR_error:.2f}")
        
        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.vstack((hz, z))

        # show animations 
        plt.cla()
        plt.plot(hz[:, 0], hz[:, 1], ".g")
        plt.plot(hxTrue[0, :].flatten(),
                 hxTrue[1, :].flatten(), "-b")
        plt.plot(hxDR[0, :].flatten(),
                 hxDR[1, :].flatten(), "-k")
        plt.plot(hxEst[0, :].flatten(),
                 hxEst[1, :].flatten(), "-r")
        EKF.plot_covariance_ellipse(xEst, PEst)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)

    assert est_error < 0.5



        

    
