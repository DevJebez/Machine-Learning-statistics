'''
Problem Statement
True Temperature: The environment temperature is set precisely at 100°C during calibration.

Sensor Readings: 
Due to noise, each sensor reading fluctuates around the true value with Gaussian noise (assumed known standard deviation, say 𝜎=1.5∘𝐶


Goal: Estimate the bias (i.e., how much the sensor is off from the true 100°C) as accurately as possible.

Constraint: You can take only 100 measurements per sensor.

Your Tasks

1. Simulate 100 sensor readings assuming the true mean is  100°𝐶+unknown bias

2. Estimate the bias using the sample mean.

3. Compute the Fisher Information and Cramer-Rao Lower Bound to quantify the best achievable accuracy.

4. Check if your empirical variance matches the theoretical CRLB.

Provide a Report stating:

1. Estimated bias

2. Expected estimation error (standard deviation from CRLB)

3. Whether the sensor is acceptable or needs recalibration.


'''


import numpy as np

#environment setup
true_temperature = 100.0  
true_bias = 0.8          
true_std = 1.5            
n_samples = 100           

# for reproducibility 
np.random.seed(123)

#generating sensor readings
readings = np.random.normal(true_temperature + true_bias, true_std, n_samples)


estimated_mean = np.mean(readings)
estimated_bias = estimated_mean - true_temperature


fisher_information = n_samples / (true_std**2)

print(f"fishers_information:{fisher_information}")

crlb = 1 / fisher_information

print("Estimated Bias (°C):", estimated_bias)
print("Cramer-Rao Lower Bound (Variance of bias estimate):", crlb)
print("Standard Deviation (Estimation Error):", np.sqrt(crlb))
