#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
from statsmodels.tsa.holtwinters import Holt 
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
from sqlalchemy import create_engine


# In[2]:


data=pd.read_csv("C:\\Program Files\\datasets\\solarpower_cumuldaybyday2.csv")
user='root'
pw='root'
db='forecasting'
engine= create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
data.to_sql("solar",con=engine,chunksize=1000,if_exists='replace',index=False)
sql="select * from solar"
df=pd.read_sql_query(sql,engine)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.tail(12)


# In[6]:


df.info()


# In[7]:


train = df.head(2000)
test= df.tail(558)


# In[8]:


def MAPE(pred, actual):
    temp = np.abs((pred - actual)/actual)*100
    return np.mean(temp)


# In[9]:


mv_pred = df["cum_power"].rolling(558).mean()
mv_pred.tail(558)
MAPE(mv_pred.tail(558), test.cum_power)


# In[10]:


df.cum_power.plot(label = "actual")
for i in range(2, 9, 2):
    df.cum_power.rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)


# In[11]:


import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(df.cum_power, lags = 4)
tsa_plots.plot_pacf(df.cum_power, lags = 4)


# In[12]:


ses_model = SimpleExpSmoothing(train["cum_power"]).fit()
pred_ses = ses_model.predict(start = test.index[0], end = test.index[-1])
ses = MAPE(pred_ses, test.cum_power) 


# In[13]:


hw_model = Holt(train["cum_power"]).fit()
pred_hw = hw_model.predict(start = test.index[0], end = test.index[-1])
hw = MAPE(pred_hw, test.cum_power)


# In[14]:


hwe_model_add_add = ExponentialSmoothing(train["cum_power"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0], end = test.index[-1])
hwe = MAPE(pred_hwe_add_add, test.cum_power) 


# In[15]:


hwe_model_mul_add = ExponentialSmoothing(train["cum_power"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0], end = test.index[-1])
hwe_w = MAPE(pred_hwe_mul_add, test.cum_power) 


# In[16]:


di = pd.Series({'Simple Exponential Method':ses, 'Holt method ':hw, 'hw_additive seasonality and additive trend':hwe, 'hw_multiplicative seasonality and additive trend':hwe_w})
mape = pd.DataFrame(di, columns=['mape'])
mape


# In[17]:


hwe_model_add_add = ExponentialSmoothing(df["cum_power"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()


# In[18]:


hwe_model_add_add.save("model.pickle")


# In[19]:


import os
os.getcwd()


# In[21]:


from statsmodels.tsa.arima.model import ARIMA
model1 = ARIMA(train.cum_power, order = (12, 1, 6))
res1 = model1.fit()
print(res1.summary())


# In[22]:


start_index = len(train)
end_index = len(train) + len(test) - 1 
forecast_test = res1.predict(start=start_index, end=end_index)


# In[23]:


print(forecast_test)


# In[24]:


from math import sqrt
from sklearn.metrics import mean_squared_error
rmse_test = sqrt(mean_squared_error(test.cum_power, forecast_test))
print('Test RMSE: %.3f' % rmse_test)


# In[25]:


from matplotlib import pyplot 
pyplot.plot(test.cum_power)
pyplot.plot(forecast_test, color = 'red')
pyplot.show()


# In[26]:


pip install pmdarima


# In[27]:


import pmdarima as pm
ar_model = pm.auto_arima(train.cum_power, start_p=0, start_q=0,
                         max_p=12, max_q=12,
                         m=12,
                         d=None,
                         seasonal=True,
                         start_P=0, trace=True,
                         error_action='warn', stepwise=True)


# In[29]:


model = ARIMA(train.cum_power, order = (2, 1, 0))
res = model.fit()
print(res.summary())


# In[31]:


start_index = len(train)
end_index = len(train) + len(test) - 1 
forecast_best = res.predict(start = start_index, end = end_index)


# In[32]:


print(forecast_best)


# In[33]:


rmse_best = sqrt(mean_squared_error(test.cum_power, forecast_best))
print('Test RMSE: %.3f' % rmse_best)
pyplot.plot(test.cum_power)
pyplot.plot(forecast_best, color = 'red')
pyplot.show()


# In[34]:


print('Test RMSE with Auto-ARIMA: %.3f' % rmse_best)
print('Test RMSE with out Auto-ARIMA: %.3f' % rmse_test)


# In[35]:


res1.save("model.pickle")


# In[36]:


from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")


# In[41]:


start_index = len(df)
end_index = len(train) + len(test) 
forecast = model.predict(start = start_index, end = end_index)


# In[42]:


print(forecast)


# In[ ]:




