import math, random
import pandas as pd
import numpy as np

def daily_reward(stocks, arm, t, amount=1.0):
    tickers = sorted(set([key.split("-")[0] for key in stocks.keys()[1:]]))
    ticker = tickers[arm]

    if ticker == "brk":
        ticker = "brk-a"

    open_price = stocks["{0}-open".format(ticker)][t]
    close_price = stocks["{0}-close".format(ticker)][t]
    return amount * (close_price / open_price - 1)

def draw(prob_dist):
    threshold = random.uniform(0, 1)

    for i, prob in enumerate(prob_dist):
        threshold -= prob
        if threshold <= 0:
            return i

def exp3(stocks, reward_func, gamma):
    K, T = len(stocks.keys()) // 2, len(stocks)
    weights = [1.0] * K

    for t in range(T):
        prob_dist = tuple((1.0 - gamma) * (w / sum(weights)) + gamma / K for w in weights)
        arm = draw(prob_dist)
        reward = reward_func(stocks, arm, t)
        weights[arm] *= math.exp(gamma * reward / K)
        yield arm, reward, weights

def exp3_P(stocks, reward_func, alpha, gamma):
    weights = [math.exp((alpha * gamma / 3) * math.sqrt(T / K))] * K

    for t in range(T):
        prob_dist = tuple((1.0 - gamma) * (w / sum(weights)) + gamma / K for w in weights)
        arm = draw(prob_dist)
        reward = reward_func(arm, t)

        for k in range(K):
            if k == arm:
                temp = (reward / prob_dist[k]) + alpha / (prob_dist[k] * math.sqrt(K * T))
            else:
                temp = alpha / (prob_dist[k] * math.sqrt(K * T))
            weights[arm] *= math.exp(temp * gamma / (3 * K)) 

        yield arm, reward, weights

if __name__ == "__main__":
    stocks = pd.read_csv("stocks/random_stocks.csv")

    for gamma in range(1, 100, 1):
        gamma /= 100
        cumulative_rewards = [0] * 25

        for i in range(25):
            for (arm, reward, weights) in exp3(stocks, daily_reward, gamma):
                cumulative_rewards[i] += reward
            
        print("{0} : {1} {2}".format(gamma, np.mean(cumulative_rewards), np.var(cumulative_rewards)))
  
    """
    K, R, T = 10, 7, 50000

    biases = [1.0 / k for k in range(2, K + 2)]
    reward_vector = [[1 if random.random() < bias else 0 for bias in biases] for _ in range(T)]
    reward_func = lambda arm, t: reward_vector[t][arm]

    best_action = max(range(K), key=lambda arm: sum([reward_vector[t][arm] for t in range(T)]))

    cumulative_reward = 0
    best_action_cumulative_reward = 0
    G = [0.0] * K
    
    for r in range(R):
        t = 0

        g = K * math.log(K) / (math.e - 1) * (4 ** r)
        gamma = min(1, math.sqrt(K * math.log(K) / ((math.e - 1) * g)))
        print(math.sqrt(K * math.log(K) / ((math.e - 1) * g)))

        for (arm, reward, weights) in exp3(K, T, reward_func, gamma):
            cumulative_reward += reward
            best_action_cumulative_reward += reward_vector[t][best_action]
            G[arm] += reward
            if max(G) >= g - K / gamma:
                print(t, max(G), g - K / gamma)
                break
            t += 1

    print("{0} : {1}".format(gamma, best_action_cumulative_reward - cumulative_reward))

    gamma = 0.05
    alpha = 1
    #for gamma in range(1, 100, 1):
    cumulative_reward = 0
    best_action_cumulative_reward = 0
    #gamma /= 100
    t = 0
        
    for (arm, reward, weights) in exp3_P(K, T, reward_func, alpha, gamma):
        cumulative_reward += reward
        best_action_cumulative_reward += reward_vector[t][best_action]
        t += 1
    
    print("{0} : {1}".format(gamma, best_action_cumulative_reward - cumulative_reward))
"""