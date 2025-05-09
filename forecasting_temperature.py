import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA 
from statsmodels.tools.eval_measures import rmse 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime 
import pandas as pd

# Load the dataset 
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
data = pd.read_csv(url, parse_dates=["Date"], index_col="Date")

print(data.head())

#visualize the dataset 
plt.figure(figsize= (15,7))
plt.plot(data)
plt.title("Teoperature data from Melbourne(1981-1980)")
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.legend()
plt.show()


# Splitting the dataset
train_size = int(len(data)*0.80)
train, test = data[train_size:], data[:train_size]

#fit the model 
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()

#forecasting the test
forecast_steps = len(test)
forecast = model_fit.forecast(steps = forecast_steps)

#evalutation metrics
print(f"Mean Absolute error:{mean_absolute_error(test,forecast)}")
print(f"Mean Squared Error:{mean_squared_error(test,forecast)}")
print(f"Root mean squared error:{rmse(test,forecast)}")

plt.figure(figsize=(15,7))
plt.plot(test.index , test, label = "Actual")
plt.plot(test.index, forecast, label = "Predicted", color = "Orange")
plt.title("Actual Vs Predicted")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.show()

future_steps = 30
future_forecast = model_fit.forecast(steps = future_steps)
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(pd.date_range(test.index[-1], periods=future_steps + 1, freq='D')[1:], 
         future_forecast, label='Forecasted', color='green')
plt.title(f'ARIMA Model - Forecast for Next {future_steps} Days')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
