import numpy as np
import numpy.typing as npt
import C.conditions as conditions

def anomaly_quadratic(charges:npt.NDArray):
    return conditions.anomaly_quadratic(charges)

def anomaly_cubic(charges:npt.NDArray):
    return conditions.anomaly_cubic(charges)

"""
def yukawa(charges_sum:npt.NDArray):
    return conditions.yukawa(charges_sum)
"""

def multiple_check(found_charges:npt.NDArray, curr_charges:npt.NDArray):
    return conditions.multiple_check(found_charges, curr_charges)