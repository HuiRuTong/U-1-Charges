from C.charges import *
from C.conditions import *
import numpy as np

def generic_rwd(found_charges, curr_charges, curr_coef, prev_coef):
    """
        Gain some reward for each coef that's 0
    """
    r = 0
    sorted_charges = get_sorted_charges(curr_charges)
    for found in found_charges:
        if multiple_check(found, sorted_charges):
            return -50, False

    for i in range(3):
        if not curr_coef[i]:
            r += 100
    if r == 300:
        found_charges.append(sorted_charges)
        return 500, True

    return r, False

def tot_improvement_rwd(found_charges, curr_charges, curr_coeff, prev_coeff):
    """
        Gain a tiny reward if the current sum
        of coef is less than before and lose
        some if more than
    """
    curr_tot_coef = np.sum(curr_coeff)
    prev_tot_coef = np.sum(prev_coeff)
    r = 0

    sorted_charges = get_sorted_charges(curr_charges)
    for found in found_charges:
        if multiple_check(found, sorted_charges):
            return -50, False

    if not curr_coeff[0] and not curr_coeff[1] and not curr_coeff[2]:
        # Im fucking stupid. The reason this was signitifcantly better was
        # because it wasn't actually finding answers where all coefs
        # were 0
        found_charges.append(sorted_charges)
        return 500, True

    if abs(curr_tot_coef) < abs(prev_tot_coef):
        r += 30

    return r, False

def split_improvement_rwd(found_charges, curr_charges, curr_coef, prev_coef):
    """
        Gain a miniscule reward if the any of the
        coef are less than before and lose some if
        any are more than
    """
    r = 0

    sorted_charges = get_sorted_charges(curr_charges)
    for found in found_charges:
        if multiple_check(found, sorted_charges):
            return -50, False

    for i in range(3):
        if not curr_coef[i]:
            r += 100
        else:
            if abs(curr_coef[i]) < abs(prev_coef[i]):
                r += 10
    if r == 300:
        found_charges.append(sorted_charges)
        return 500, True
        
    return r, False

def abs_err_rwd(found_charges, curr_charges, curr_coef, prev_coef):
    """
        Basically just split_improvement reward but w/o
        a fixed reward
    """
    r = 0
    is_valid = 1
    
    sorted_charges = get_sorted_charges(curr_charges)
    for found in found_charges:
        if multiple_check(found, sorted_charges):
            return -5, False

    for i in range(3):
        if curr_coef[i] == 0:
            r += 5
        else:
            is_valid = 0
            r += (-curr_coef[i] + prev_coef[i]) / 10**i

    if is_valid:
        found_charges.append(sorted_charges)
        return r, True

    return r, False