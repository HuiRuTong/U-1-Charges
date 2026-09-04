import numpy as np
import gymnasium as gym
from src.charge_space import Charge_Space
from C.conditions import *
from C.charges import *

class Charge_Env(gym.Env):
    def __init__(self, max_charge=5, max_steps=15, rwd_func=None, seed=None):
        self.max_charge = max_charge
        self.max_steps = max_steps
        self._np_random_seed = seed
        self._np_random = np.random.default_rng(seed)

        self.observation_space = Charge_Space(max_charge=5)
        self.action_space = gym.spaces.Dict(
                                {
                                 "choose_particle":gym.spaces.Discrete(6, dtype=np.int32),
                                 "choose_generation":gym.spaces.Discrete(3, dtype=np.int32),
                                 "modify_charge":gym.spaces.Discrete(2, dtype=np.int32),
                                }
                            )   # Mostly just here for convention's sake. Goes unused
        self.charges, self.charges_sum, self.curr_coef, self.prev_coef = self.observation_space.sample()

        self.rwd_func = rwd_func
        self.rewards_sum = 0.0
        self.steps = 0

        # Here for reference. Unused
        # self.particles = {"q":0, "l":1, "e":2, "u":3, "d":4, "v":5}
        # self.actions = {"choose_particle":0, "choose_generation":1, "choose_mod":2}

    def _upd_charges(self):
        """
            Helper function to modify the 3rd charges of each particle
            and coefficients after a change is made.
        """
        self.charges_sum = get_charges_properties(self.charges)

        for i in range(2, 6): 
            self.charges[i, 2] = self.charges_sum[i] - self.charges[i, 0] - self.charges[i, 1]

        for i in range(3):
            self.prev_coef[i] = self.curr_coef[i]

        self.curr_coef[0] = anomaly_quadratic(self.charges)
        self.curr_coef[1] = anomaly_cubic(self.charges)
        self.curr_coef[2] = yukawa(self.charges_sum)

    def _log_charges(self, log_file):
        """
            Helper function to record each found charge inside a text file

            log_file :
                File where found charges are stored
        """
        for i in range(6):
            log_file.write(f"  {self.charges[i, 0]: }  {self.charges[i, 1]: }  {self.charges[i, 2]: }")
        log_file.write('\n')

    def _get_obs(self):
        return self.charges

    def _get_info(self):
        return f"quadratic coef: {self.curr_coef[0]}\ncubic coef: {self.curr_coef[1]}\nyukawa coef: {self.curr_coef[2]}"

    def step(self, action, found_charges, log_file=None):
        chosen_particle = action[0].item()
        chosen_generation = action[1].item()
        chosen_mod = action[2].item()

        self.charges[chosen_particle, chosen_generation] += chosen_mod + (-1 if not chosen_mod else 1)
        self._upd_charges()

        reward, terminated = self.rwd_func(found_charges, self.charges, self.curr_coef, self.prev_coef)
        truncated = False
        if terminated and log_file is not None:
            self._log_charges(log_file)

        self.steps += 1
        if (self.steps > self.max_steps):
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def reset(self, seed=0):
        super().reset(seed=seed)

        self.charges, self.charges_sum, self.curr_coef, self.prev_coef = self.observation_space.sample()

        return self._get_obs(), self._get_info()