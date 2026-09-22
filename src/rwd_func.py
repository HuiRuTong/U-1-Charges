from C.charges import *
from C.conditions import *
import numpy as np

def gen_rwd(found_charges, curr_charges, curr_coef, prev_coef):
    """
        Gain some reward for each coef that's 0
    """
    r = 0
    is_valid = 1

    sorted_charges = get_sorted_charges(curr_charges)
    """for found in found_charges:
        if multiple_check(found, sorted_charges):
            return -50, False"""

    for i in range(3):
        if curr_coef[i]:
            is_valid = 0
        else:
            r += 5

    if is_valid:
        found_charges.append(sorted_charges)
        return 15, True

    return r, False

def tot_improvement_rwd(found_charges, curr_charges, curr_coef, prev_coef):
    """
        Gain a reward if the current sum
        of coef is less than before and lose
        some if more than
    """
    is_valid = 1
    curr_tot_coef = 0
    prev_tot_coef = 0

    sorted_charges = get_sorted_charges(curr_charges)

    for i in range(3):
        if curr_coef[i]:
            is_valid = 0

        curr_tot_coef += np.abs(curr_coef[i])
        prev_tot_coef += np.abs(prev_coef[i])
    
    if is_valid:
        found_charges.append(sorted_charges)
        return 15, True

    return 9 if curr_tot_coef < prev_tot_coef else -9, False

def split_improvement_rwd(found_charges, curr_charges, curr_coef, prev_coef):
    """
        Gain a miniscule reward if the any of the
        coef are less than before and lose some if
        any are more than
    """
    r = 0
    is_valid = 1

    sorted_charges = get_sorted_charges(curr_charges)
    for i in range(3):
        if curr_coef[i]:
            is_valid = 0
            if abs(curr_coef[i]) < abs(prev_coef[i]):
                r += 3

    if is_valid:
        found_charges.append(sorted_charges)
        return 15, True
        
    return r, False

def abs_tot_err_rwd(found_charges, curr_charges, curr_coef, prev_coef):
    is_valid = 1
    curr_tot_coef = 0
    prev_tot_coef = 0

    sorted_charges = get_sorted_charges(curr_charges)

    for i in range(3):
        if curr_coef[i]:
            is_valid = 0

        curr_tot_coef += np.abs(curr_coef[i] / 10**(i+1))
        prev_tot_coef += np.abs(prev_coef[i] / 10**(i+1))

    if is_valid:
        found_charges.append(sorted_charges)
        return 15, True

    return -curr_tot_coef + prev_tot_coef, False

def abs_err_rwd(found_charges, curr_charges, curr_coef, prev_coef):
    """
        Basically just split_improvement reward but w/o
        a fixed reward
    """
    r = 0
    is_valid = 1
    
    sorted_charges = get_sorted_charges(curr_charges)

    for i in range(3):
        if curr_coef[i]:
            is_valid = 0
            r += (-abs(curr_coef[i]) + abs(prev_coef[i])) / 10**(i+1)

    if is_valid:
        found_charges.append(sorted_charges)
        return 15, True

    return r, False