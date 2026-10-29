from C.charges import *
import numpy as np

def gen_rwd(found_charges, curr_charges, curr_coef, prev_coef, min_err, gae_gamma):
    """
        Gain some reward for each coef that's 0
    """
    r = 0
    is_valid = 1

    for i in range(3):
        if curr_coef[i]:
            is_valid = 0
        else:
            r += 5

    if is_valid:
        found_charges.add(curr_charges)
        return 15, True

    return r, False

def tot_improvement_rwd(found_charges, curr_charges, curr_coef, prev_coef, min_err, gae_gamma):
    """
        Gain a reward if the current sum
        of coef is less than before and lose
        some if more than
    """
    is_valid = 1
    curr_tot_coef = 0
    prev_tot_coef = 0

    for i in range(3):
        if curr_coef[i]:
            is_valid = 0

        curr_tot_coef += np.abs(curr_coef[i])
        prev_tot_coef += np.abs(prev_coef[i])
    
    if is_valid:
        found_charges.add(curr_charges)
        return 15, True

    return 9 if curr_tot_coef < prev_tot_coef else -9, False

def split_improvement_rwd(found_charges, curr_charges, curr_coef, prev_coef, min_err, gae_gamma):
    """
        Gain a miniscule reward if the any of the
        coef are less than before and lose some if
        any are more than
    """
    r = 0
    is_valid = 1
    
    for i in range(3):
        if curr_coef[i]:
            is_valid = 0
            if abs(curr_coef[i]) < abs(prev_coef[i]):
                r += 3

    if is_valid:
        found_charges.add(curr_charges)
        return 15, True
        
    return r, False

def abs_tot_err_rwd(found_charges, curr_charges, curr_coef, prev_coef, min_err, gae_gamma):
    is_valid = 1
    curr_tot_coef = 0
    prev_tot_coef = 0

    for i in range(3):
        if curr_coef[i]:
            is_valid = 0

        curr_tot_coef += np.abs(curr_coef[i] / 10**(i+1))
        prev_tot_coef += np.abs(prev_coef[i] / 10**(i+1))

    if is_valid:
        found_charges.add(curr_charges)
        return 15, True

    return -curr_tot_coef + prev_tot_coef, False

def abs_err_rwd(found_charges, curr_charges, curr_coef, prev_coef, min_err, gae_gamma):
    """
        Basically just split_improvement reward but w/o
        a fixed reward
    """
    r = 0
    is_valid = 1
    
    for i in range(3):
        if curr_coef[i]:
            is_valid = 0
        
        r += (-abs(curr_coef[i]) + abs(prev_coef[i])) / 10**(i+1)

    if is_valid:
        found_charges.add(curr_charges)
        return 15, True

    return r, False

def min_err_rwd(found_charges, curr_charges, curr_coef, prev_coef, min_err, gae_gamma):
    """
        Rewards the agent if it manages to surpass its previous best
        state

        This is the only function so far that returns differently.
        I'd really like to avoid recomputing the weighted err, and w/o pointers,
        this ugly ass solution is the only one I could think of
    """
    r = 0
    is_valid = 1
    err = 0
    
    for i in range(3):
        if curr_coef[i]:
            is_valid = 0
            err += abs(curr_coef[i]) / 10**(i+1)

    if is_valid:
        found_charges.add(curr_charges)
        return 15, True, 0

    if err < min_err:
        return 0.1, False, err
    return 0, False, min_err

def potential_rwd(found_charges, curr_charges, curr_coef, prev_coef, min_err, gae_gamma):
    """
        Did you say POTENTIAL?

        No but actually i need to find a better way to add these args,
        or else there'll be 100 of them by the time the agent starts learning
    """

    is_valid = 1
    curr_potential = 0
    prev_potential = 0

    for i in range(3):
        if curr_coef[i]:
            is_valid = 0

        curr_potential -= np.abs(curr_coef[i]) / (np.abs(curr_coef[i]) + 10**(i+1))
        prev_potential -= np.abs(prev_coef[i]) / (np.abs(prev_coef[i]) + 10**(i+1))

    if is_valid:
        found_charges.add(curr_charges)

    return 15*is_valid + 5*(gae_gamma*curr_potential - prev_potential), is_valid