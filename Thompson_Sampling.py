#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import shuffle
import pandas as pd
import numpy as np


# In[2]:


np.random.beta([81,90],[279,100])


# In[3]:


fortune_500 = pd.read_csv("stocks/top_10_market_cap.csv")
#random_stocks = pd.read_csv("stocks/random-stocks.csv")


# In[ ]:


test = list(fortune_500.keys()[1:])
test1 = [name.split('-')[0] for name in test]
test1 = list(set(test1))


# In[ ]:


fortune_500


# In[ ]:


fortune_500.keys()


# In[28]:


def single_day_reward(stocktable, name, day, amountToInvest = 1.0):
    if name == "brk":
        name = "brk-a"
    openPrice = stocktable[name+"-open"][day]
    closePrice = stocktable[name+"-close"][day]
    sharesBought = amountToInvest / openPrice
    amountAfterSale = sharesBought * closePrice
 
    return amountAfterSale - amountToInvest


# In[13]:


fortune_500.keys()


# In[15]:


def best_action_reward(stocks,  iterations, amountToInvest = 1.0):
    max_reward = -100000
    best_stock = ""
    stocks_columns = list(stocks.keys()[1:])
    stock_names = [name.split('-')[0] for name in stocks_columns]
    stock_names = list(set(stock_names))
    for name in stock_names:
        reward = 0
        if name == "brk":
            name = "brk-a"
        for i in range(0,iterations):
            reward += single_day_reward(stocks, name, i)
        if reward > max_reward:
            max_reward = reward
            best_stock = name
    return max_reward, best_stock


# In[26]:


fortune_500


# In[ ]:





# In[20]:


def random(stocks,iterations):
    cumulative_reward = [0]
    best_action_cumulative_reward = [0]
    #Alpha = np.ones(9)
    #Beta = np.ones(9)
    stocks_columns = list(stocks.keys()[1:])
    stock_names = [name.split('-')[0] for name in stocks_columns]
    stock_names = list(set(stock_names))
    for i in range(0,iterations):
        max_idx = np.random.randint(low = 0, high = len(stock_names))
        name = stock_names[max_idx]
        if name == "brk":
            name = "brk-a"
        reward = single_day_reward(stocks, name, i)
        cumulative_reward.append(cumulative_reward[i]+reward)
    return cumulative_reward, best_action_reward(stocks,  iterations)
    


# In[21]:


def Thomason_Sampling(stocks,iterations):
    cumulative_reward = [0]
    best_action_cumulative_reward = [0]
    Alpha = np.ones(9)
    Beta = np.ones(9)
    stocks_columns = list(stocks.keys()[1:])
    stock_names = [name.split('-')[0] for name in stocks_columns]
    stock_names = list(set(stock_names))
    for i in range(0,iterations):
        thetas = np.random.beta(Alpha, Beta)
        max_idx = np.argmax(thetas)
        name = stock_names[max_idx]
        if name == "brk":
            name = "brk-a"
        reward = single_day_reward(stocks, name, i)
        Alpha[max_idx] += reward
        Beta[max_idx] += 1-reward
        cumulative_reward.append(cumulative_reward[i]+reward)
    return cumulative_reward, best_action_reward(stocks,  iterations), Alpha, Beta
    
    


# In[29]:


cr, bcr = random(fortune_500, 2518)


# In[31]:


cr_thom, bcr_thom, a,b = Thomason_Sampling(fortune_500,2518)


# In[35]:


bcr


# In[33]:


cr_thom[2517]


# In[34]:


cr[2517]


# In[ ]:


bcr[2606]


# In[ ]:


cr[1000]


# In[ ]:


bcr[1000]


# In[ ]:


def payoff(stockTable, t, stock, amountToInvest=1.0):
   openPrice, closePrice = stockTable[stock][t]
 
   sharesBought = amountToInvest / openPrice
   amountAfterSale = sharesBought * closePrice
 
   return amountAfterSale - amountToInvest


# In[ ]:




