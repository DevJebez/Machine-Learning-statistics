import numpy as np
import matplotlib.pyplot as plt 

true_mean = 5.0
true_std = 2.0 

# generate gaussian distributed samples 

# for reproducibility
np.random.seed(0)

number_of_samples = 10000
data = np.random.normal(true_mean, true_std, number_of_samples)
print(data)

estimated_mean = np.mean(data)

#Fisher's information for the estimated parameter - mean 
fisher_information = number_of_samples / (true_std ** 2)
print(f"fishers information : {fisher_information}")
# calculate variance of estimated mean 

crlb = 1/ fisher_information
print("True mean :" , true_mean)
print("Estimated mean :", estimated_mean)
print("Cramer Rao Lower Bound(variance of estimated mean)", crlb)

plt.figure(figsize=(10,6))
plt.hist(data, bins = 30 , label="generated data")
plt.axvline(true_mean,color = 'red', label = f"True mean:{true_mean}")
plt.axvline(estimated_mean,color = 'green', label = f"True mean:{estimated_mean}")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()