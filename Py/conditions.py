import numpy as np
import numpy.typing as npt

def anomaly_check_quadratic(charges:npt.NDArray):
    return np.sum(charges[0,:]**2 + charges[2,:]**2 - 2*charges[3,:]**2
                  - charges[4,:]**2 + charges[5,:]**2)

def anomaly_check_cubic(charges:npt.NDArray):
    return np.sum(6*charges[0,:]**3 - charges[1,:]**3 - charges[2,:]**3
                  - 3*charges[3,:]**3 + 2*charges[4,:]**3 - 3*charges[5,:]**3)

def yukawa(charges_sum:npt.NDArray):
    q_sum = charges_sum[0]
    v_sum = charges_sum[1]
    e_sum = charges_sum[2]
    u_sum = charges_sum[3]
    l_sum = charges_sum[4]
    d_sum = charges_sum[5]

    return (-q_sum - v_sum - e_sum - u_sum + l_sum - d_sum)

def multiple_check(found_charges:npt.NDArray, curr_charges:npt.NDArray):
    # Assumes charges are already sorted in order of increasing magnitude
    dot = 0
    a_sqr = 0
    b_sqr = 0
    for i in range(18):
        dot += found_charges[i] * curr_charges[i]
        a_sqr += found_charges[i] * found_charges[i]
        b_sqr += curr_charges[i] * curr_charges[i]

    if (dot*dot == a_sqr*b_sqr):
        return True
    return False