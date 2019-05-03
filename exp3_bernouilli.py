import numpy as np

def draw(prob_dist):
    threshold = np.random.uniform(0, 1)

    for i, prob in enumerate(prob_dist):
        threshold -= prob
        if threshold <= 0:
            return i

def exp3(K, T, reward_func, gamma):
    weights = [1.0] * K

    for t in range(T):
        prob_dist = tuple((1.0 - gamma) * (w / sum(weights)) + gamma / K for w in weights)
        arm = draw(prob_dist)
        reward = reward_func(arm, t)
        weights[arm] *= np.exp(gamma * reward / K)
        yield arm, reward, weights

def exp3_P(K, T, reward_func, alpha, gamma):
    K, T = len(stocks.keys()) // 2, len(stocks)
    weights = [np.exp((alpha * gamma / 3) * np.sqrt(T / K))] * K

    for t in range(T):
        prob_dist = tuple((1.0 - gamma) * (w / sum(weights)) + gamma / K for w in weights)
        arm = draw(prob_dist)
        reward = reward_func(arm, t)
        for k in range(K):
            update = alpha / (prob_dist[k] * np.sqrt(K * T))
            update = (reward / prob_disk[k]) + update if k == arm else update
            weights[arm] *= np.exp(update * gamma / (3 * K))
        yield arm, reward, weights

if __name__ == "__main__":
    T = 5000
    
    p_vector = [0.5, 0.25]
    reward_vector = [[1 if np.random.random() < p else 0 for p in p_vector] for _ in range(T)]
    reward_func = lambda arm, t: reward_vector[t][arm]

    K = len(p_vector)
    
    for gamma in np.arange(0.01, 1.01, 0.01):
        cumulative_reward, optimal_reward, t = 0, 0, 0 
        
        for (arm, reward, weights) in exp3(K, T, reward_func, gamma):
            cumulative_reward += reward
            optimal_reward += np.amax(reward_vector[t])
            t += 1

        print("{0:.2f} : {1} {2}".format(gamma, optimal_reward - cumulative_reward, weights))
