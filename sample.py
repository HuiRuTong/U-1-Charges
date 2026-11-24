from src.charge_space import *
import numpy as np

space = Charge_Space(5)
num_samples = 10000

coefs = []
errs = []    # i.e. the sum of charges
for i in range(num_samples):
    sample_coef = space.sample()[2]

    coefs.append(np.abs(sample_coef))
    errs.append(np.sum(np.divide(np.abs(sample_coef), np.array([10, 100]))))

coefs = np.array(np.array(coefs))
errs = np.array(np.array(errs))

mean_coef = np.mean(coefs, axis=0)
mean_err = np.mean(errs)

median_coef_scale = np.median(coefs // 10, axis=0)
median_err_scale = np.median(errs // 10, axis=0)

print(mean_coef, mean_err, median_coef_scale, median_err_scale)