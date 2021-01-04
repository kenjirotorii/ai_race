import math
import numpy as np


def state_transition(pose, omega, vel, dt):
    theta0 = pose[2]
    if math.fabs(omega) < 1e-10:
        return pose + np.array([vel*math.cos(theta0), vel*math.sin(theta0), omega]) * dt
    else:
        return pose + np.array([vel/omega*(math.sin(theta0 + omega*dt) - math.sin(theta0)), 
                                vel/omega*(-math.cos(theta0 + omega*dt) + math.cos(theta0)), 
                                omega*dt])

def distance_from_centerline(pose):
    x = pose[0]
    y = pose[1]

    scaler = 6.0 / 1120
    ref_x = 120*scaler
    ref_y = 232*scaler
    ref_r = 208*scaler

    if x >= ref_x and y >= ref_y:
        return math.fabs(np.sqrt((x - ref_x)**2 + (y - ref_y)**2) - ref_r)
    elif x <= -ref_x and y >= ref_y:
        return math.fabs(np.sqrt((x + ref_x)**2 + (y - ref_y)**2) - ref_r)
    elif x <= -ref_x and y <= -ref_y:
        return math.fabs(np.sqrt((x + ref_x)**2 + (y + ref_y)**2) - ref_r)
    elif x >= ref_x and y <= -ref_y:
        return math.fabs(np.sqrt((x - ref_x)**2 + (y + ref_y)**2) - ref_r)
    elif x >= -ref_x and x <= ref_x and y >= 0 and y >= x + 112*scaler and y >= - x + 112*scaler:
        return math.fabs(y - 440*scaler)
    elif x >= -ref_x and x <= ref_x and y <= 0 and y <= x - 112*scaler and y <= - x - 112*scaler:
        return math.fabs(y + 440*scaler)
    elif y >= -ref_y and y <= ref_y and x >= 0 and y <= x + 112*scaler and y >= - x - 112*scaler:
        return math.fabs(x - 328*scaler)
    elif y >= -ref_y and y <= ref_y and x <= 0 and y <= - x + 112*scaler and y >= x - 112*scaler:
        return math.fabs(x + 328*scaler)
    else:
        return None


def distance_from_inline(pose):
    x = pose[0]
    y = pose[1]
    
    scaler = 6.0 / 750

    ref_xm = -70*scaler
    ref_xp = 60*scaler
    ref_ym = -138*scaler
    ref_yp = 138*scaler
    ref_r = 100*scaler

    if x >= ref_xp and y >= ref_yp:
        return np.sqrt((x - ref_xp)**2 + (y - ref_yp)**2) - ref_r
    elif x <= ref_xm and y >= ref_yp:
        return np.sqrt((x - ref_xm)**2 + (y - ref_yp)**2) - ref_r
    elif x <= ref_xm and y <= ref_ym:
        return np.sqrt((x - ref_xm)**2 + (y - ref_ym)**2) - ref_r
    elif x >= ref_xp and y <= ref_ym:
        return np.sqrt((x - ref_xp)**2 + (y - ref_ym)**2) - ref_r
    elif x >= ref_xm and x <= ref_xp and y >= 0 and y >= x + 78*scaler and y >= -x + 68*scaler:
        return y - 238*scaler
    elif x >= ref_xm and x <= ref_xp and y <= 0 and y <= -x - 78*scaler and y <= x - 68*scaler:
        return -y - 238*scaler
    elif y >= ref_ym and y <= ref_yp and x >= -5*scaler and y <= x + 78*scaler and y >= -x - 78*scaler:
        return x - 160*scaler
    elif y >= ref_ym and y <= ref_yp and x <= -5*scaler and y <= -x + 68*scaler and y >= x - 68*scaler:
        return -x - 160*scaler
    else:
        return None

