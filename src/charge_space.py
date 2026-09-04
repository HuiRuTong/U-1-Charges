import numpy as np
import gymnasium as gym
from C.conditions import *
from C.charges import *

class Charge_Space(gym.spaces.Box):
    def __init__(self, max_charge=5, charges=np.zeros((6,3))):
        super().__init__(low=-max_charge, high=max_charge, shape = (6, 3), dtype = np.int32)
        self.max_charge = max_charge

    def _get_ordered_charges(self, n):
        """
            Helper function to generate charges in order of descending magnitude.
            
            Quite frankly, this isn't very useful since the order gets messed up
            during training anyway.

            n :
                number of charges to generate
        """

        particle_charges = []
        bounds = self.max_charge

        for i in range(n):
            particle_charges.append(self.np_random.integers(-np.abs(bounds), np.abs(bounds), dtype=np.int32, endpoint=True))
            bounds = particle_charges[i]

        return np.array(particle_charges, dtype=np.int32)

    def sample(self):
        """
            Samples a random set of charges that are in order of descending magnitude for each particle
        """
        while(1):
            q_charges = self._get_ordered_charges(3)
            v_charges = self._get_ordered_charges(3)
            l_charges = np.concat(
                        (self._get_ordered_charges(2),
                         np.zeros((1,), dtype=np.int32)), axis=0)
            e_charges = np.concat(
                        (self._get_ordered_charges(2),
                         np.zeros((1,), dtype=np.int32)), axis=0)
            u_charges = np.concat(
                        (self._get_ordered_charges(2),
                         np.zeros((1,), dtype=np.int32)), axis=0)
            d_charges = np.concat(
                        (self._get_ordered_charges(2),
                         np.zeros((1,), dtype=np.int32)), axis=0)

            charges = np.stack((q_charges, v_charges, e_charges, u_charges, l_charges, d_charges), axis=0)
            charges_sum = get_charges_properties(charges)

            for i in range(1, 6): 
                charges[i, 2] = charges_sum[i] - charges[i, 0] - charges[i, 1]
                if (np.abs(charges[i, 2]) > np.abs(charges[i, 1])):
                    break
            else:
                break

        curr_quad, curr_cube, curr_yukawa = anomaly_quadratic(charges), anomaly_cubic(charges), yukawa(charges_sum)
        curr_coef = np.array([curr_quad, curr_cube, curr_yukawa])

        return charges, charges_sum, curr_coef, np.zeros((3,))

    def contains(self, sample):
        if not super().contains(sample):
            return False

        for i in range(6):
            if (np.abs(sample[i, 1]) > np.abs(sample[i, 0]) or np.abs(sample[i, 2]) > np.abs(sample[i, 1])):
                return False