#!/usr/bin/env python
# coding: utf-8

# In[207]:


from random import shuffle
import pandas as pd
import numpy as np
import math
import scipy
import scipy.stats
import matplotlib.pyplot as plt


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





# In[23]:


def reward_vector(num_arms):
    return np.linspace(0.1,0.9,num_arms)


# In[18]:


rv = reward_vector(2)
def reward_stochastic(choice, reward_vector):
    return np.random.binomial(1,reward_vector[choice],1)[0]


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
    
    


# In[82]:


np.insert(np.ones(4),0,2)


# In[295]:


regret_ucbs_mean[10000]


# In[299]:


class Experiment:
    def __init__(self, iterations, num_arms):
        self.reward_vector= reward_vector(num_arms)
        self.iterations = iterations
        self.num_arms = num_arms
        self.adversarial_reward_matrix = np.zeros((2,iterations))
        for i in range(0,iterations):
            if i < 1000:
                self.adversarial_reward_matrix[0,i] = 1
            else:
                self.adversarial_reward_matrix[1,i] = 1
        self.reward_matrix = np.zeros((num_arms, iterations))
        self.cumulative_adversarial_reward_per_arm = np.zeros((2, iterations))
        for i in range(0,2):
            for j in range(0,iterations):
                self.cumulative_adversarial_reward_per_arm[i,j] = np.sum(self.adversarial_reward_matrix[i,0:j+1])
        self.best_cumulative_adversarial_reward_per_round = np.insert(np.max(self.cumulative_adversarial_reward_per_arm,axis = 0),0,0)
        for i in range(0,num_arms):
            for j in range(0,iterations):
                self.reward_matrix[i,j] = np.random.binomial(1,self.reward_vector[i],1)[0]
        self.cumulative_reward_per_arm = np.zeros((num_arms, iterations))
        for i in range(0,num_arms):
            for j in range(0,iterations):
                self.cumulative_reward_per_arm[i,j] = np.sum(self.reward_matrix[i,0:j+1])
        self.best_cumulative_reward_per_round = np.insert(np.max(self.cumulative_reward_per_arm,axis = 0),0,0)
    def random(self):
        cumulative_reward = [0]
        for i in range(0,self.iterations):
            choice = np.random.randint(0,self.num_arms)
            reward = self.reward_matrix[choice, i]
            cumulative_reward.append(cumulative_reward[i]+reward)
        return np.array(cumulative_reward), self.best_cumulative_reward_per_round - np.array(cumulative_reward)
    def Thomason_Sampling_adversarial(self):
        cumulative_reward = [0]
        Alpha = np.ones(2)
        Beta = np.ones(2)
        for i in range(0,self.iterations):
            thetas = np.random.beta(Alpha, Beta)
            max_idx = np.argmax(thetas)
            reward = self.adversarial_reward_matrix[max_idx, i]
            Alpha[max_idx] += reward
            Beta[max_idx] += 1-reward
            cumulative_reward.append(cumulative_reward[i]+reward)
        return np.array(cumulative_reward), self.best_cumulative_adversarial_reward_per_round - np.array(cumulative_reward)
    def UCB_adversarial(self):
        cumulative_reward = [0]
        num_actions = np.zeros(2)
        total_reward = np.zeros(2)
        average_reward = np.zeros(2)
        # Play each action once
        for i in range(0,self.iterations):
            if (i < 2):
                reward = self.adversarial_reward_matrix[i,i]
                cumulative_reward.append(cumulative_reward[i]+reward)
                average_reward[i] += reward
                total_reward[i] += reward
                num_actions[i] += 1
            else:
                temp = average_reward + np.sqrt(2*math.log(i)/num_actions)
                max_idx = np.argmax(temp)
                reward = self.adversarial_reward_matrix[max_idx, i]
                cumulative_reward.append(cumulative_reward[i]+reward)
                total_reward[max_idx] += reward
                num_actions[max_idx] += 1
                average_reward = total_reward / num_actions
        return np.array(cumulative_reward), self.best_cumulative_adversarial_reward_per_round - np.array(cumulative_reward)
    def random_adversarial(self):
        cumulative_reward = [0]
        for i in range(0,self.iterations):
            choice = np.random.randint(0,2)
            reward = self.adversarial_reward_matrix[choice, i]
            cumulative_reward.append(cumulative_reward[i]+reward)
        return np.array(cumulative_reward), self.best_cumulative_adversarial_reward_per_round - np.array(cumulative_reward)
    def Thomason_Sampling(self):
        cumulative_reward = [0]
        Alpha = np.ones(self.num_arms)
        Beta = np.ones(self.num_arms)
        for i in range(0,self.iterations):
            thetas = np.random.beta(Alpha, Beta)
            max_idx = np.argmax(thetas)
            reward = self.reward_matrix[max_idx, i]
            Alpha[max_idx] += reward
            Beta[max_idx] += 1-reward
            cumulative_reward.append(cumulative_reward[i]+reward)
        return np.array(cumulative_reward), self.best_cumulative_reward_per_round - np.array(cumulative_reward)
    def UCB(self):
        cumulative_reward = [0]
        num_actions = np.zeros(self.num_arms)
        total_reward = np.zeros(self.num_arms)
        average_reward = np.zeros(self.num_arms)
        # Play each action once
        for i in range(0,self.iterations):
            if (i < self.num_arms):
                reward = self.reward_matrix[i,i]
                cumulative_reward.append(cumulative_reward[i]+reward)
                average_reward[i] += reward
                total_reward[i] += reward
                num_actions[i] += 1
            else:
                temp = average_reward + np.sqrt(2*math.log(i)/num_actions)
                max_idx = np.argmax(temp)
                reward = self.reward_matrix[max_idx, i]
                cumulative_reward.append(cumulative_reward[i]+reward)
                total_reward[max_idx] += reward
                num_actions[max_idx] += 1
                average_reward = total_reward / num_actions
        return np.array(cumulative_reward), self.best_cumulative_reward_per_round - np.array(cumulative_reward)
            
    


# In[306]:


adverarial_regret_ucbs


# In[301]:


adverarial_regret_ucbs = []
adverarial_regret_randoms = []
adverarial_regret_Thomsons = []
for i in range(0,5):
    exp = Experiment(10000, 20)
    reward, regret = exp.random_adversarial()
    r,re = exp.Thomason_Sampling_adversarial()
    r_ucb, regret_ucb = exp.UCB_adversarial()
    adverarial_regret_ucbs.append(regret_ucb)
    adverarial_regret_randoms.append(regret)
    adverarial_regret_Thomsons.append(re)


# In[303]:


adverarial_regret_ucbs = np.array(adverarial_regret_ucbs)
adverarial_regret_randoms = np.array(adverarial_regret_randoms)
adverarial_regret_Thomsons = np.array(adverarial_regret_Thomsons)


# In[269]:


regret_ucbs = []
regret_randoms = []
regret_Thomsons = []
for i in range(0,100):
    exp = Experiment(10000, 20)
    reward, regret = exp.random()
    r,re = exp.Thomason_Sampling()
    r_ucb, regret_ucb = exp.UCB()
    regret_ucbs.append(regret_ucb)
    regret_randoms.append(regret)
    regret_Thomsons.append(re)


# In[270]:


regret_ucbs = np.array(regret_ucbs)
regret_randoms = np.array(regret_randoms)
regret_Thomsons = np.array(regret_Thomsons)


# In[271]:


regret_Thomsons_mean = np.mean(regret_Thomsons, axis = 0)
regret_ucbs_mean = np.mean(regret_ucbs, axis = 0)


# In[285]:


upper_thomson_conf = []
lower_thomson_conf = []
for i in range(0,10001):
    m,l,h = mean_confidence_interval(regret_Thomsons[:,i])
    upper_thomson_conf.append(h)
    lower_thomson_conf.append(l)


# In[286]:


upper_ucb_conf = []
lower_ucb_conf = []
for i in range(0,10001):
    m,l,h = mean_confidence_interval(regret_ucbs[:,i])
    upper_ucb_conf.append(h)
    lower_ucb_conf.append(l)


# In[238]:


regret_ucbs


# In[217]:


np.sum(regret_ucbs < 0)


# In[262]:


regret_randoms_mean = np.mean(regret_randoms, axis = 0)


# In[288]:


t


# In[255]:


regret_ucbs = np.array(regret_ucbs)
regret_ucbs_std = np.std(regret_ucbs,axis=0)
regret_ucbs_mean = np.mean(regret_ucbs, axis = 0)


# In[240]:


def mean_confidence_interval(data, confidence=0.90):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


# In[184]:


upper_bounds = [math.sqrt(20*i*math.log2(i)) for i in range(1,10001)]
lower_bounds = [0]+[math.sqrt(i*20) / 20 for i in range(1,10001)]


# In[222]:


regret_ucbs[:,5000]


# In[221]:


mean_confidence_interval(regret_ucbs[:,5000])


# In[289]:


regret_Thomsons_mean


# In[291]:


plt.semilogy(indices,regret_ucbs_mean, label = "Mean Pseudo-Regret for UCB with 99% Confidence Interval")
plt.semilogy(indices,[0] + upper_bounds, label = "Upper Bound for UCB and Thompson Pseudo-Regret")
plt.semilogy(indices,upper_ucb_conf, 'k--',  color = 'yellow')
plt.semilogy(indices,lower_ucb_conf, 'k--',  color = 'yellow')
plt.semilogy(indices,upper_thomson_conf, 'k--')
plt.semilogy(indices,lower_thomson_conf, 'k--')
plt.semilogy(indices, lower_bounds, label = "Lower Bound for UCB and Thompson Pseudo-Regret")
plt.semilogy(indices, regret_Thomsons_mean, label = "Mean Pseudo_Regret for Thompson with 99% Confidence Interval")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
plt.xlabel('Number of Rounds')
plt.ylabel('Pseudo Regret')
plt.title('UCB and Thompson Pseudo-Regret Bounds')


# In[ ]:


np.random


# In[212]:


indices = np.array([i for i in range(0,10001)])
plt.plot(indices, re)


# In[144]:


plt.plot(indices, regret)


# In[ ]:


def random(num_arms,iterations):
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


Thom

