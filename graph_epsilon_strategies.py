import matplotlib.pyplot as plt
import epsilon_strategies_bernoulli
import numpy as np
import scipy.stats as ss


def compute_confidence_intervals(pseudo_regret_vec, confidence=0.99):
    num_rounds = len(pseudo_regret_vec[0])

    mean_pseudo_regret_vec = [0]
    min_pseudo_regret_vec = [0]
    max_pseudo_regret_vec = [0]
    for t in np.arange(num_rounds):
        temp = np.array([x[t] for x in pseudo_regret_vec])
        h = ss.sem(temp) * ss.t.ppf((1 + confidence) / 2, len(temp) - 1)
        mean = np.mean(temp)
        mean_pseudo_regret_vec.append(mean)
        min_pseudo_regret_vec.append(mean - h)
        max_pseudo_regret_vec.append(mean + h)

    return mean_pseudo_regret_vec, min_pseudo_regret_vec, max_pseudo_regret_vec


def epsilon_greedy(bernoulli_params, ROUNDS, epsilon):
    cum_pseudo_regret = 0
    cum_pseudo_regret_at_round = []

    for round_reward, best_possible_round_reward in epsilon_strategies_bernoulli.epsilon_greedy(bernoulli_params,
                                                                                                ROUNDS, epsilon, ROUNDS,
                                                                                                1):
        cum_pseudo_regret += best_possible_round_reward - round_reward
        cum_pseudo_regret_at_round.append(cum_pseudo_regret)

    return np.array(cum_pseudo_regret_at_round)


def epsilon_first(bernoulli_params, ROUNDS, epsilon):
    cum_pseudo_regret = 0
    cum_pseudo_regret_at_round = []

    for round_reward, best_possible_round_reward in epsilon_strategies_bernoulli.epsilon_first(bernoulli_params, ROUNDS, epsilon):
        cum_pseudo_regret += best_possible_round_reward - round_reward
        cum_pseudo_regret_at_round.append(cum_pseudo_regret)

    return np.array(cum_pseudo_regret_at_round)

def epsilon_decreasing(bernoulli_params, ROUNDS, epsilon, N, decreasing_factor):
    cum_pseudo_regret = 0
    cum_pseudo_regret_at_round = []

    for round_reward, best_possible_round_reward in epsilon_strategies_bernoulli.epsilon_greedy(bernoulli_params,
                                                                                                    ROUNDS, epsilon,
                                                                                                    N, decreasing_factor):
        cum_pseudo_regret += best_possible_round_reward - round_reward
        cum_pseudo_regret_at_round.append(cum_pseudo_regret)

    return np.array(cum_pseudo_regret_at_round)

def epsilon_decreasing_auer(bernoulli_params, num_rounds, epsilon_init):
    cum_pseudo_regret = 0
    cum_pseudo_regret_at_round = []

    for round_reward, best_possible_round_reward in epsilon_strategies_bernoulli.epsilon_decreasing_auer(bernoulli_params, num_rounds, epsilon_init):
        cum_pseudo_regret += best_possible_round_reward - round_reward
        cum_pseudo_regret_at_round.append(cum_pseudo_regret)

    return np.array(cum_pseudo_regret_at_round)


if __name__ == "__main__":
    ROUNDS = 10000
    NUM_TRIALS = 100
    K = 20
    EPSILONS = [0, 0.25, 0.50, 0.75, 1]

    rounds_arr = np.arange(ROUNDS+1)

    plt_colors = ['red', 'green', 'blue', 'orange', 'plum']

    bernoulli_params = np.linspace(0.1, 0.9, 20)

    minimax_lower_bound = (1 / 20) * np.sqrt(K * rounds_arr)

    # plt.plot(rounds_arr, rounds_arr, color='black', label='Linear upper bound')
    # plt.plot(rounds_arr, minimax_lower_bound, color='gray', label="Minimax lower bound")
    # #Epsilon-Greedy
    # for idx, epsilon in enumerate(EPSILONS):
    #     pseudo_regret_vec = []
    #     for trials in range(NUM_TRIALS):
    #         pseudo_regret_vec.append(epsilon_greedy(bernoulli_params, ROUNDS, epsilon))
    #
    #     mean_pseudo_regret_vec, min_pseudo_regret_vec, max_pseudo_regret_vec = compute_confidence_intervals(pseudo_regret_vec)
    #
    #     plt.plot(rounds_arr, min_pseudo_regret_vec, "--", color=plt_colors[idx], linewidth=1.0)
    #     plt.plot(rounds_arr, max_pseudo_regret_vec, "--", color=plt_colors[idx], linewidth=1.0)
    #     plt.plot(rounds_arr, mean_pseudo_regret_vec, color=plt_colors[idx], label="$\epsilon$=" + str(epsilon) + " (w/ 99% CI)")
    #
    #
    # plt.title('Round vs. Cumulative Pseudo-Regret for Epsilon-Greedy')
    # plt.xlabel('Number of Rounds')
    # plt.ylabel('Cumulative Pseudo-Regret')
    # plt.yscale('log')
    # plt.legend(loc='lower right')
    # plt.show()
    #
    # # Epsilon-First
    # plt.plot(rounds_arr, rounds_arr, color='black', label='Linear upper bound')
    # plt.plot(rounds_arr, minimax_lower_bound, color='gray', label="Minimax lower bound")
    # for idx, epsilon in enumerate(EPSILONS):
    #     pseudo_regret_vec = []
    #     for trials in range(NUM_TRIALS):
    #         pseudo_regret_vec.append(epsilon_first(bernoulli_params, ROUNDS, epsilon))
    #
    #     mean_pseudo_regret_vec, min_pseudo_regret_vec, max_pseudo_regret_vec = compute_confidence_intervals(pseudo_regret_vec)
    #
    #     plt.plot(rounds_arr, min_pseudo_regret_vec, "--", color=plt_colors[idx], linewidth=1.0)
    #     plt.plot(rounds_arr, max_pseudo_regret_vec, "--", color=plt_colors[idx], linewidth=1.0)
    #     plt.plot(rounds_arr, mean_pseudo_regret_vec, color=plt_colors[idx], label="$\epsilon$=" + str(epsilon) + " (w/ 99% CI)")
    #
    # plt.title('Round vs. Cumulative Pseudo-Regret for Epsilon-First')
    # plt.xlabel('Number of Rounds')
    # plt.ylabel('Cumulative Pseudo-Regret')
    # plt.yscale('log')
    # plt.legend(loc='lower right')
    # plt.show()

    # Epsilon-Decreasing
    epsilon = 0.5
    DECREASING_FACTORS = [0.2, 0.8]
    N = 1000
    plt.plot(rounds_arr, 1000 * np.log(rounds_arr + 0.00000001), color='black', label='Logarithmic upper bound')
    plt.plot(rounds_arr, minimax_lower_bound, color='gray', label="Minimax lower bound")
    for idx, decreasing_factor in enumerate(DECREASING_FACTORS):
        pseudo_regret_vec = []
        for trials in range(NUM_TRIALS):
            pseudo_regret_vec.append(epsilon_decreasing(bernoulli_params, ROUNDS, epsilon, N, decreasing_factor))

        mean_pseudo_regret_vec, min_pseudo_regret_vec, max_pseudo_regret_vec = compute_confidence_intervals(pseudo_regret_vec)

        plt.plot(rounds_arr, min_pseudo_regret_vec, "--", color=plt_colors[idx], linewidth=1.0)
        plt.plot(rounds_arr, max_pseudo_regret_vec, "--", color=plt_colors[idx], linewidth=1.0)
        plt.plot(rounds_arr, mean_pseudo_regret_vec, color=plt_colors[idx], label="$\epsilon$=0.5, $\delta$=" + str(decreasing_factor) + ", N=1000 (w/ 99% CI)")

    # Epsilon decreasing Auer
    AUER_EPSILON = 200
    pseudo_regret_vec = []
    for trials in range(NUM_TRIALS):
        pseudo_regret_vec.append(epsilon_decreasing_auer(bernoulli_params, ROUNDS, AUER_EPSILON))
    mean_pseudo_regret_vec, min_pseudo_regret_vec, max_pseudo_regret_vec = compute_confidence_intervals(
        pseudo_regret_vec)
    plt.plot(rounds_arr, min_pseudo_regret_vec, "--", color=plt_colors[idx+1], linewidth=1.0)
    plt.plot(rounds_arr, max_pseudo_regret_vec, "--", color=plt_colors[idx+1], linewidth=1.0)
    plt.plot(rounds_arr, mean_pseudo_regret_vec, color=plt_colors[idx+1],
             label="Auer et. al. w/ $\epsilon$=200 (w/ 99% CI)")

    plt.title('Round vs. Cumulative Pseudo-Regret for Epsilon-Decreasing')
    plt.xlabel('Number of Rounds')
    plt.ylabel('Cumulative Pseudo-Regret')
    plt.yscale('log')
    plt.legend(loc='lower right')
    plt.show()

    # VDBE-epsilon
    # DELTA = 1 / K
    # INVERSE_SENSITIVITY = 1.5
    # cum_pseudo_regret = 0
    # cum_pseudo_regret_at_round = []
    #
    # for round_reward, best_possible_round_reward in epsilon_strategies_bernoulli.VDBE_epsilon_greedy(bernoulli_params, ROUNDS, DELTA, INVERSE_SENSITIVITY):
    #     cum_pseudo_regret += best_possible_round_reward - round_reward
    #     cum_pseudo_regret_at_round.append(cum_pseudo_regret)
    #
    # plt.plot(rounds_arr[:-1], cum_pseudo_regret_at_round)
    #
    # plt.title('Round vs. Cumulative Regret for VDBE-epsilon')
    # plt.xlabel('Round')
    # plt.ylabel('Cumulative Regret')
    # plt.legend(loc='upper left')
    # plt.show()
