#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import statsmodels.stats.descriptivestats as sd
from statsmodels.stats import weightstats as stests


# In[2]:


cutlet=pd.read_csv("C:\Program Files\datasets\Cutlets.csv")


# In[3]:


cutlet.info()


# In[4]:


cutlet.shape


# In[5]:


cutlet.isnull().sum()


# In[10]:


cutlet.columns


# In[31]:


cutlet=cutlet.iloc[:35,:]


# In[32]:


print(stats.shapiro(cutlet['Unit A']))


# In[33]:


print(stats.shapiro(cutlet['Unit B']))


# In[34]:


mean_a=np.mean(cutlet['Unit A'])
mean_a


# In[35]:


mean_b= np.mean(cutlet['Unit B'])
mean_b


# In[40]:


ztest, pval = stests.ztest(cutlet['Unit A'], x2 = None, value = mean_a)
print(float(pval))


# In[44]:


ztest, pval = stests.ztest(cutlet['Unit A'], x2 = None, value = mean_a, alternative = 'larger')
print(float(pval))


# In[45]:


lab=pd.read_csv("C:\Program Files\datasets\lab_tat_updated.csv")


# In[46]:


lab.info()


# In[47]:


lab.shape


# In[48]:


lab_mean_a=np.mean(lab['Laboratory_1'])
lab_mean_a


# In[49]:


lab_mean_b=np.mean(lab['Laboratory_2'])
lab_mean_b


# In[50]:


lab_mean_c=np.mean(lab['Laboratory_3'])
lab_mean_c


# In[51]:


lab_mean_d=np.mean(lab['Laboratory_4'])
lab_mean_d


# In[53]:


stats.shapiro(lab.Laboratory_1)
stats.shapiro(lab.Laboratory_2)
stats.shapiro(lab.Laboratory_3)
stats.shapiro(lab.Laboratory_4)


# In[54]:


scipy.stats.levene(lab.Laboratory_1, lab.Laboratory_2, lab.Laboratory_3,lab.Laboratory_4)


# In[55]:


F, p = stats.f_oneway(lab.Laboratory_1, lab.Laboratory_2, lab.Laboratory_3,lab.Laboratory_4)
p


# In[56]:


buy_ratio=pd.read_csv("C:\Program Files\datasets\BuyerRatio.csv")


# In[57]:


contingency_table = pd.DataFrame({
    'East': [50, 550],
    'West': [142, 351],
    'North': [131, 480],
    'South': [70, 350]
}, index=['Males', 'Females'])


# In[59]:


from scipy.stats import chi2_contingency
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)


# In[60]:


alpha = 0.05
print(f'Chi-square Statistic: {chi2_stat}')
print(f'P-Value: {p_value}')


# In[61]:


if p_value < alpha:
    print('Reject the null hypothesis. Not all proportions are equal.')
else:
    print('Fail to reject the null hypothesis. All proportions are equal.')


# In[62]:


tele=pd.read_csv("C:\Program Files\datasets\CustomerOrderform.csv")


# In[63]:


tele.shape


# In[64]:


tele.columns


# In[71]:


tele.Phillippines.value_counts()


# In[72]:


tele.Indonesia.value_counts()


# In[73]:


tele.Malta.value_counts()


# In[74]:


tele.India.value_counts()


# In[76]:


obs=np.array([[271,267,269,280],[29,33,31,20]])


# In[78]:


Chisquares_result=scipy.stats.chi2_contingency(obs)
Chisquares_result


# In[81]:


Chi_square = [['Test Statistic', 'p-value'], [Chisquares_result[0], Chisquares_result[1]]]
Chi_square


# In[82]:


data=pd.read_csv("C:\Program Files\datasets\Fantaloons.csv")


# In[86]:


data.Weekdays.value_counts()


# In[87]:


data.Weekend.value_counts()


# In[88]:


count=np.array([[287,233],[113,167]])


# In[89]:


result=scipy.stats.chi2_contingency(count)
result


# In[90]:


Chi_square_2 = [['Test Statistic', 'p-value'], [result[0], result[1]]]
Chi_square_2


# In[ ]:




