import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_process import ArmaProcess

#To generate synthetic data we define parameters for ARMA model
ar_params = np.array([0.65,-0.2])
ma_params = np.array([0.75,0.30])

#Convert into arma format
ar = np.r_[1,-ar_params]
ma = np.r_[1, ma_params]

arma_process = ArmaProcess(ar,ma)
data = arma_process.generate_sample(nsample=500)

#plotting the data using plot function 
plt.figure(figsize=(15,7))
plt.plot(data, label ="Synthetic generated data")
plt.title("Synthetic data")
plt.xlabel("Time")
plt.ylabel("Value")

plt.show()


# plotting the acf and pacf function
fig , axes = plt.subplots(1,2, figsize = (15,7))
plot_acf(data, lags = 30,ax= axes[0])
axes[0].set_title("Auto Correlation function")

plot_pacf(data, lags= 30, ax = axes[1])
axes[1].set_title("Partial Autocorrelation function")

plt.show()


# we do adf test to check the stationarity

adf_result = adfuller(data)
print("ADF Statistic :", adf_result[0])
print("p-value:", adf_result[1])

for key, value in adf_result[4].items():
    print(f"Critical value {key}:{value}")

if(adf_result[1] < 0.05):
    print("The time series is stationary (Reject H0)")
else:
    print("The time series is not stationary(Fail to reject H0)")