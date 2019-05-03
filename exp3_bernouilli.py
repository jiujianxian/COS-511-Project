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

def plot_regret_vs_time(K, T, reward_func, gamma, reward_vec, optimal_arm, num_rounds):
    pseudo_regret_vec = []
    for r in np.arange(num_rounds):
        cumulative_reward, optimal_reward, t = 0, 0, 0
        pseudo_regret_round_vec = [0]
        for (arm, reward, weights) in exp3(K, T, reward_func, gamma):
            cumulative_reward += reward
            optimal_reward += reward_vec[t][optimal_arm]
            pseudo_regret_round_vec.append(optimal_reward - cumulative_reward)
            t += 1
        pseudo_regret_vec.append(pseudo_regret_round_vec)

    min_pseudo_regret_vec = [0]
    max_pseudo_regret_vec = [0]
    for t in np.arange(T):
        temp = np.array([x[t] for x in pseudo_regret_vec])
        h = ss.sem(temp) * ss.t.ppf((1 + 0.99) / 2, len(temp) - 1)
        min_pseudo_regret_vec.append(np.mean(temp) - h)
        max_pseudo_regret_vec.append(np.mean(temp) + h)

    pyplot.figure()
    pyplot.plot("Pseudo-Regret of Exp3, Gamma = {0}".format(gamma))
    pyplot.plot(np.arange(T + 1), (np.e - 1) * gamma * np.arange(T + 1) + (K * np.log(K)) / gamma, "red", label="Upper Bound")
    pyplot.plot(np.arange(T + 1), (1 / 20) * np.sqrt(K * np.arange(T + 1)), "green", label="Lower Bound")    
    pyplot.plot(np.arange(T + 1), min_pseudo_regret_vec, "b--", linewidth=1.0)
    pyplot.plot(np.arange(T + 1), max_pseudo_regret_vec, "b--", linewidth=1.0)
    pyplot.plot(np.arange(T + 1), np.mean(pseudo_regret_vec, axis=0), "blue", label="Mean")
    pyplot.xlabel("Round, $T$")
    pyplot.ylabl("Pseudo-Regret")
    pyplot.legend()
    pyplot.show()

def plot_regret_vs_time_P(K, T, reward_func, alpha, gamma, reward_vec, optimal_arm):
    cumulative_reward, optimal_reward, t = 0, 0, 0
    pseudo_regret_vec = [0]

    for (arm, reward, weights) in exp3_P(K, T, reward_func, alpha, gamma):
        cumulative_reward += reward
        optimal_reward += reward_vec[t][optimal_arm]
        pseudo_regret_vec.append(optimal_reward - cumulative_reward)
        t += 1

    pyplot.figure()
    pyplot.plot(np.arange(T+1), pseudo_regret_vec, "blue")
    pyplot.plot(np.arange(T+1), (np.e - 1) * gamma * np.arange(T + 1) + (K * np.log(K)) / gamma, "red")
    pyplot.plot(np.arange(T+1), (1 / 20) * np.sqrt(K * np.arange(T+1)), "green")
    pyplot.show()

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    gamma = 0.07
    alpha = 8.2
    gamma_vec = np.arange(0.05, 0.15, 0.01)
    p_vec = np.linspace(0.1, 0.9, num=20)
    K, T, num_rounds = len(p_vec), 1000, 10

    reward_vec = [[np.random.binomial(1, p) for p in p_vec] for _ in range(T)] 
    reward_func = lambda t, arm: reward_vec[t][arm]
    optimal_arm = max(range(K), key=lambda arm: sum([reward_vec[t][arm] for t in range(T)]))

    plot_regret_vs_time(K, T, reward_func, gamma, reward_vec, optimal_arm, num_rounds)
    #plot_regret_vs_time_P(K, T, reward_func, alpha, gamma, reward_vec, optimal_arm)
    """
    pseudo_regret_vec = []

    for gamma in gamma_vec:
        pseudo_regret_round_vec = []
        for r in np.arange(num_rounds):
            best_reward, cumulative_reward, t = 0, 0, 0
            for (arm, reward, weights) in exp3(K, T, reward_func, gamma):
                best_reward += reward_vec[t][optimal_arm]
                cumulative_reward += reward
                t += 1
            pseudo_regret_round_vec.append(best_reward - cumulative_reward)
        pseudo_regret_vec.append(pseudo_regret_round_vec)
        print("{0:.2f} : {1}".format(gamma, pseudo_regret_round_vec))

    pseudo_regret_vec = np.mean(pseudo_regret_vec, axis=1)

    pyplot.figure()
    pyplot.plot(gamma_vec, np.array(pseudo_regret_vec))
    pyplot.show()    
    """
