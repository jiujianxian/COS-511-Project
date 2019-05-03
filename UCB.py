#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import shuffle
import pandas as pd
import numpy as np


# In[2]:


fortune_500 = pd.read_csv("stocks/top_10_market_cap.csv")


# In[3]:


def single_day_reward(stocktable, name, day, amountToInvest = 1.0):
    if name == "brk":
        name = "brk-a"
    openPrice = stocktable[name+"-open"][day]
    closePrice = stocktable[name+"-close"][day]
    sharesBought = amountToInvest / openPrice
    amountAfterSale = sharesBought * closePrice
 
    return amountAfterSale - amountToInvest


# In[4]:


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


# In[11]:


def ucb1(stocks, iterations):
    cumulative_reward = [0]
    best_action_cumulative_reward = 0
    stocks_columns = list(stocks.keys()[1:])
    stock_names = [name.split('-')[0] for name in stocks_columns]
    stock_names = list(set(stock_names))
    num_actions = np.zeros(len(stock_names))
    total_reward = np.zeros(len(stock_names))
    average_reward = np.zeros(len(stock_names))
    # Play each action once
    for i in range(0,iterations):
        if (i < len(stock_names)):
            name = stock_names[i]
            if name == "brk":
                name = "brk-a"
            reward = single_day_reward(stocks, name, i)
            cumulative_reward.append(cumulative_reward[i]+reward)
            average_reward[i] += reward
            total_reward[i] += reward
            num_actions[i] += 1
        else:
            max_idx = np.argmax(average_reward)
            name = stock_names[max_idx]
            if name == "brk":
                name = "brk-a"
            reward = single_day_reward(stocks, name, i)
            cumulative_reward.append(cumulative_reward[i]+reward)
            total_reward[max_idx] += reward
            num_actions[max_idx] += 1
            average_reward = total_reward / num_actions
    return cumulative_reward, best_action_reward(stocks,  iterations)
            


# In[13]:


cr, bcr = ucb1(fortune_500, 2518)


# In[15]:


cr[2517]


# In[16]:


bcr


# In[ ]:




