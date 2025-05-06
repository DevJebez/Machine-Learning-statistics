import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

ar_params  = np.array([0.6, -0.3])
ma_params = np.array([0.5,0.2])
ar = np.r_[1,-ar_params]
ma = np.r_[1,ma_params]

arma_process = ArmaProcess(ar, ma)
data = arma_process.generate_sample(nsample=300)

plt.figure(figsize=(12,5))
plt.plot(data, label ="Synthetic ARMA data", color = 'blue')
plt.title("Synthetic data generated from ARMA model")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()


fig, axes = plt.subplots(1,2, figsize=(14,7))

plot_acf(data, lags= 30,ax = axes[0])

axes[0].set_title("Auto Correlation Function")

plot_pacf(data, lags = 30, ax = axes[1])

axes[1].set_title("Partial Auto Correlation function")

plt.show()