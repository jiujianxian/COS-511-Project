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

def plot_regret_vs_time(K, T, reward_func, reward_vec, optimal_arm, num_rounds):
    gamma = min(1, np.sqrt(K * np.log(K) / ((np.e - 1) * T)))
    
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
    
    print("Exp3 : {0}".format(np.mean(pseudo_regret_vec, axis=0)[-1]))
    print("Exp3 : {0}".format(np.var(pseudo_regret_vec, axis=0)[-1]))

    pyplot.figure()
    pyplot.title("Round vs. Cumulative Pseudo-Regret for Exp3, $\\gamma$ = {0:.2f}".format(gamma))
    pyplot.plot(np.arange(T + 1), (np.e - 1) * gamma * np.arange(T + 1) + (K * np.log(K)) / gamma, "black", label="Upper bound")
    pyplot.plot(np.arange(T + 1), (1 / 20) * np.sqrt(K * np.arange(T + 1)), "grey", label="Minimax lower bound")    
    pyplot.plot(np.arange(T + 1), min_pseudo_regret_vec, "b--", linewidth=1.0)
    pyplot.plot(np.arange(T + 1), max_pseudo_regret_vec, "b--", linewidth=1.0)
    pyplot.plot(np.arange(T + 1), np.mean(pseudo_regret_vec, axis=0), "blue", label="Exp3 (w/ 99% CI)")
    pyplot.xlabel("Number of Rounds")
    pyplot.ylabel("Cumulative Pseudo-Regret")
    pyplot.legend()
    pyplot.yscale("log")
    pyplot.show()

def plot_regret_vs_time_P(K, T, reward_func, reward_vec, optimal_arm, num_rounds):
    delta = 0.01
    alpha = 2 * np.sqrt(np.log(K * T / delta))
    gamma = min(3 / 5, 2 * np.sqrt(3 * K * np.log(K) / (5 * T)))
    
    pseudo_regret_vec = []
    for r in np.arange(num_rounds):
        cumulative_reward, optimal_reward, t = 0, 0, 0
        pseudo_regret_round_vec = [0]
        for (arm, reward, weights) in exp3_P(K, T, reward_func, alpha, gamma):
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

    print("Exp3.P : {0}".format(np.mean(pseudo_regret_vec, axis=0)[-1]))
    print("Exp3.P : {0}".format(np.var(pseudo_regret_vec, axis=0)[-1]))

    pyplot.figure()
    pyplot.title("Round vs. Cumulative Pseudo-Regret for Exp3.P, $\\alpha$ = {0:.2f}, $\\gamma$ = {1:.2f}".format(alpha, gamma))
    pyplot.plot(np.arange(T + 1), 4 * np.sqrt(K * np.arange(T + 1) * np.log(K * np.arange(T + 1) / delta)) + 4 * np.sqrt(5 * K * np.arange(T + 1) * np.log(K) / 3) + 8 * np.log(K * np.arange(T + 1) / delta), "black", label="Upper bound")
    pyplot.plot(np.arange(T + 1), (1 / 20) * np.sqrt(K * np.arange(T + 1)), "grey", label="Minimax lower bound")    
    pyplot.plot(np.arange(T + 1), min_pseudo_regret_vec, "b--", linewidth=1.0)
    pyplot.plot(np.arange(T + 1), max_pseudo_regret_vec, "b--", linewidth=1.0)
    pyplot.plot(np.arange(T + 1), np.mean(pseudo_regret_vec, axis=0), "blue", label="Exp3.P (w/ 99% CI)")
    pyplot.xlabel("Number of Rounds")
    pyplot.ylabel("Cumulative Pseudo-Regret")
    pyplot.legend()
    pyplot.yscale("log")
    pyplot.show()

def plot_regret_vs_gamma(K, T, reward_func, reward_vec, optimal_arm, num_rounds):
    gamma_vec = np.arange(0.01, 1.01, 0.01)
    
    pseudo_regret_vec = []
    for gamma in gamma_vec:
        pseudo_regret_round_vec = []
        for r in np.arange(num_rounds):
            cumulative_reward, optimal_reward, t = 0, 0, 0
            for (arm, reward, weights) in exp3(K, T, reward_func, gamma):
                cumulative_reward += reward
                optimal_reward += reward_vec[t][optimal_arm]
                t += 1
            pseudo_regret_round_vec.append(optimal_reward - cumulative_reward)
        pseudo_regret_vec.append(pseudo_regret_round_vec)
        print("{0:.2f} : {1}".format(gamma, np.mean(pseudo_regret_round_vec)))

    min_pseudo_regret_vec = []
    max_pseudo_regret_vec = []
    for i, _ in enumerate(gamma_vec):
        temp = np.array([x for x in pseudo_regret_vec[i]])
        h = ss.sem(temp) * ss.t.ppf((1 + 0.99) / 2, len(temp) - 1)
        min_pseudo_regret_vec.append(np.mean(temp) - h)
        max_pseudo_regret_vec.append(np.mean(temp) + h)
    
    pyplot.figure()
    pyplot.title("Pseudo-Regret of Exp3 vs Gamma")
    pyplot.plot(gamma_vec, np.mean(pseudo_regret_vec, axis=1), "blue", label="Exp3 (w/ 99% CI)")
    pyplot.plot(gamma_vec, min_pseudo_regret_vec, "b--", linewidth=1.0)
    pyplot.plot(gamma_vec, max_pseudo_regret_vec, "b--", linewidth=1.0)
    pyplot.xlabel("Gamma, $\\gamma$")
    pyplot.ylabel("Pseudo-Regret, $\\bar{R}_{n}$")
    pyplot.legend()
    pyplot.show()    

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    p_vec = np.linspace(0.1, 0.9, num=20)
    K, T, num_rounds = len(p_vec), 10000, 100

    reward_vec = [[np.random.binomial(1, p) for p in p_vec] for _ in range(T)] 
    reward_func = lambda t, arm: reward_vec[t][arm]
    optimal_arm = max(range(K), key=lambda arm: sum([reward_vec[t][arm] for t in range(T)]))

    plot_regret_vs_time(K, T, reward_func, reward_vec, optimal_arm, num_rounds)
    plot_regret_vs_time_P(K, T, reward_func, reward_vec, optimal_arm, num_rounds)
    plot_regret_vs_gamma(K, T, reward_func, reward_vec, optimal_arm, num_rounds)
