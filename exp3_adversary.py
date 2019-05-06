import numpy as np
import scipy.stats as ss
from matplotlib import pyplot

def draw(prob_dist):
    threshold = np.random.uniform(0, 1)
    for i, prob in enumerate(prob_dist):
        threshold -= prob
        if threshold <= 0:
            return i

def exp3(K, T, reward_func, gamma):
    weights = np.array([1.0] * K)
    for t in range(T):
        prob_dist = [(1.0 - gamma) * (w / np.sum(weights)) + gamma / K for w in weights]
        arm = draw(prob_dist)
        reward = reward_func(t, arm)
        weights[arm] *= np.exp(gamma * reward / (K * prob_dist[arm]))
        weights /= np.sum(weights)
        yield arm, reward, weights

def exp3_P(K, T, reward_func, alpha, gamma):
    weights = np.array([np.exp((alpha * gamma / 3) * np.sqrt(T / K))] * K)
    for t in range(T):
        prob_dist = [(1.0 - gamma) * (w / np.sum(weights)) + gamma / K for w in weights]
        arm = draw(prob_dist)
        reward = reward_func(t, arm)
        for k in range(K):
            update = alpha / (prob_dist[k] * np.sqrt(K * T))
            update = (reward / prob_dist[k]) + update if k == arm else update
            weights[arm] *= np.exp(update * gamma / (3 * K))
            weights /= np.sum(weights)
        yield arm, reward, weights

def plot_regret_vs_time(K, T, num_rounds):
    gamma = min(1, np.sqrt(K * np.log(K) / ((np.e - 1) * T)))
    
    pseudo_regret_vec = []
    optimal_reward = 0
    for r in np.arange(num_rounds):
        cumulative_reward, t = 0, 0
        optimal_rewards, reward_vec = [0] * K, [1, 1]
        pseudo_regret_round_vec = []
        for (arm, reward, weights) in exp3(K, T, lambda t, arm: reward_vec[arm], gamma):
            cumulative_reward += reward
            for k in np.arange(K):
                optimal_rewards[k] += reward_vec[k]
            pseudo_regret_round_vec.append([optimal_rewards[k] - cumulative_reward for k in np.arange(K)])
            if (arm == 0 and reward == 0) or (arm == 1 and reward == 1):
                reward_vec = [1, 0]
            elif (arm == 1 and reward == 0) or (arm == 0 and reward == 1):
                reward_vec = [0, 1]
            t += 1
        optimal_arm = np.argmax(pseudo_regret_round_vec[-1])
        pseudo_regret_round_vec = [x[optimal_arm] for x in pseudo_regret_round_vec]
        pseudo_regret_vec.append(pseudo_regret_round_vec)
        optimal_reward += optimal_rewards[optimal_arm]

    min_pseudo_regret_vec = []
    max_pseudo_regret_vec = []
    for t in np.arange(T):
        temp = np.array([x[t] for x in pseudo_regret_vec])
        h = ss.sem(temp) * ss.t.ppf((1 + 0.99) / 2, len(temp) - 1)
        min_pseudo_regret_vec.append(np.mean(temp) - h)
        max_pseudo_regret_vec.append(np.mean(temp) + h)
    
    print("Exp3 : {0}".format(np.mean(pseudo_regret_vec, axis=0)[-1]))
    print("Exp3 : {0}".format(np.var(pseudo_regret_vec, axis=0)[-1]))

    pyplot.figure()
    pyplot.title("Round vs. Cumulative Pseudo-Regret of Exp3, $\\gamma$ = {0:.2f}".format(gamma))
    pyplot.plot(np.arange(T), min_pseudo_regret_vec, "b--", linewidth=1.0)
    pyplot.plot(np.arange(T), max_pseudo_regret_vec, "b--", linewidth=1.0)
    pyplot.plot(np.arange(T), np.mean(pseudo_regret_vec, axis=0), "blue", label="Exp3 (w/ 99% CI)")
    pyplot.xlabel("Number of Rounds")
    pyplot.ylabel("Cumulative Pseudo-Regret")
    pyplot.legend()
    pyplot.show()

def plot_regret_vs_time_P(K, T, num_rounds):
    delta = 0.01
    alpha = 2 * np.sqrt(np.log(K * T / delta))
    gamma = min(3 / 5, 2 * np.sqrt(3 * K * np.log(K) / (5 * T)))
    
    pseudo_regret_vec = []
    optimal_reward = 0
    for r in np.arange(num_rounds):
        cumulative_reward, t = 0, 0
        optimal_rewards, reward_vec = [0] * K, [1, 1]
        pseudo_regret_round_vec = []
        for (arm, reward, weights) in exp3_P(K, T, lambda t, arm: reward_vec[arm], alpha, gamma):
            cumulative_reward += reward 
            for k in np.arange(K):
                optimal_rewards[k] += reward_vec[k]
            pseudo_regret_round_vec.append([optimal_rewards[k] - cumulative_reward for k in np.arange(K)]) 
            if (arm == 0 and reward == 0) or (arm == 1 and reward == 1):
                reward_vec = [1, 0]
            elif (arm == 1 and reward == 0) or (arm == 0 and reward == 1):
                reward_vec = [0, 1]
            t += 1
        optimal_arm = np.argmax(pseudo_regret_round_vec[-1])
        pseudo_regret_round_vec = [x[optimal_arm] for x in pseudo_regret_round_vec]
        pseudo_regret_vec.append(pseudo_regret_round_vec)
        optimal_reward += optimal_rewards[optimal_arm]

    min_pseudo_regret_vec = []
    max_pseudo_regret_vec = []
    for t in np.arange(T):
        temp = np.array([x[t] for x in pseudo_regret_vec])
        h = ss.sem(temp) * ss.t.ppf((1 + 0.99) / 2, len(temp) - 1)
        min_pseudo_regret_vec.append(np.mean(temp) - h)
        max_pseudo_regret_vec.append(np.mean(temp) + h)

    print("Exp3.P : {0}".format(np.mean(pseudo_regret_vec, axis=0)[-1]))
    print("Exp3.P : {0}".format(np.var(pseudo_regret_vec, axis=0)[-1]))

    pyplot.figure()
    pyplot.title("Round vs. Cumulative Pseudo-Regret of Exp3.P, $\\alpha$ = {0:.2f}, $\\gamma$ = {1:.2f}".format(alpha, gamma))
    pyplot.plot(np.arange(T), min_pseudo_regret_vec, "b--", linewidth=1.0)
    pyplot.plot(np.arange(T), max_pseudo_regret_vec, "b--", linewidth=1.0)
    pyplot.plot(np.arange(T), np.mean(pseudo_regret_vec, axis=0), "blue", label="Exp3.P (w/ 99% CI)")
    pyplot.xlabel("Number of Rounds")
    pyplot.ylabel("Cumulative Pseudo-Regret")
    pyplot.legend()
    pyplot.show()

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
 
    K, T, num_rounds = 2, 10000, 10

    plot_regret_vs_time(K, T, num_rounds)
    plot_regret_vs_time_P(K, T, num_rounds)
