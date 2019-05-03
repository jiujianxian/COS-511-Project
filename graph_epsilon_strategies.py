import matplotlib.pyplot as plt
import epsilon_strategies_bernoulli
import pandas as pd
import numpy as np

def epsilon_greedy(bernoulli_params, ROUNDS, epsilon):
    cum_regret = 0
    cum_regret_at_round = []

    for round_reward, best_possible_round_reward in epsilon_strategies_bernoulli.epsilon_greedy(bernoulli_params,
                                                                                                ROUNDS, epsilon, ROUNDS,
                                                                                                1):
        cum_regret += best_possible_round_reward - round_reward
        cum_regret_at_round.append(cum_regret)

    return np.array(cum_regret_at_round)


def epsilon_first(bernoulli_params, ROUNDS, epsilon):
    cum_regret = 0
    cum_regret_at_round = []

    for round_reward, best_possible_round_reward in epsilon_strategies_bernoulli.epsilon_first(bernoulli_params, ROUNDS, epsilon):
        cum_regret += best_possible_round_reward - round_reward
        cum_regret_at_round.append(cum_regret)

    return np.array(cum_regret_at_round)

def epsilon_decreasing(bernoulli_params, ROUNDS, epsilon, N, decreasing_factor):
    cum_regret = 0
    cum_regret_at_round = []

    for round_reward, best_possible_round_reward in epsilon_strategies_bernoulli.epsilon_greedy(bernoulli_params,
                                                                                                    ROUNDS, epsilon,
                                                                                                    N, decreasing_factor):
        cum_regret += best_possible_round_reward - round_reward
        cum_regret_at_round.append(cum_regret)

    return np.array(cum_regret_at_round)


if __name__ == "__main__":
    stocks = pd.read_csv("stocks/random_stocks.csv")
    tickers = sorted(set([key.split("-")[0] for key in stocks.keys()[1:]]))
    ROUNDS = 10000
    NUM_TRIALS = 100
    EPSILONS = [0, 0.25, 0.50, 0.75, 1]

    rounds_arr = list(range(ROUNDS))

    bernoulli_params = np.linspace(0.1, 0.9, 20)
    # Epsilon-Greedy
    # for epsilon in EPSILONS:
    #     avg_cum_regret_at_round = np.zeros((ROUNDS))
    #     for trials in range(NUM_TRIALS):
    #         avg_cum_regret_at_round += epsilon_greedy(bernoulli_params, ROUNDS, epsilon)
    #
    #     avg_cum_regret_at_round /= ROUNDS
    #
    #     plt.plot(rounds_arr, avg_cum_regret_at_round, label='epsilon=' + str(epsilon))
    #
    # plt.title('Round vs. Cumulative Regret for Epsilon-Greedy')
    # plt.xlabel('Round')
    # plt.ylabel('Cumulative Regret')
    # plt.yscale('log')
    # plt.legend(loc='upper left')
    # plt.show()

    # Epsilon-First
    # for epsilon in EPSILONS:
    #     avg_cum_regret_at_round = np.zeros((ROUNDS))
    #     for trials in range(NUM_TRIALS):
    #         avg_cum_regret_at_round += epsilon_first(bernoulli_params, ROUNDS, epsilon)
    #
    #     avg_cum_regret_at_round /= ROUNDS
    #
    #     plt.plot(rounds_arr, avg_cum_regret_at_round, label='epsilon=' + str(epsilon))
    #
    # plt.title('Round vs. Cumulative Regret for Epsilon-First')
    # plt.xlabel('Round')
    # plt.ylabel('Cumulative Regret')
    # plt.yscale('log')
    # plt.legend(loc='upper left')
    # plt.show()

    # Epsilon-Decreasing
    epsilon = 0.5
    DECREASING_FACTORS = [0, 0.2, 0.4, 0.6, 0.8]
    N = 1000
    for decreasing_factor in DECREASING_FACTORS:
        avg_cum_regret_at_round = np.zeros((ROUNDS))
        for trials in range(NUM_TRIALS):
            avg_cum_regret_at_round += epsilon_decreasing(bernoulli_params, ROUNDS, epsilon, N, decreasing_factor)

        avg_cum_regret_at_round /= ROUNDS

        plt.plot(rounds_arr, avg_cum_regret_at_round, label='delta=' + str(decreasing_factor))

    plt.title('Round vs. Cumulative Regret for Epsilon-Decreasing')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.show()

    # VDBE-epsilon
    DELTA = 1 / len(tickers)
    INVERSE_SENSITIVITY = 1.5
    cum_regret = 0
    cum_regret_at_round = []

    for round_reward, best_possible_round_reward in epsilon_strategies_bernoulli.VDBE_epsilon_greedy(bernoulli_params, ROUNDS, DELTA, INVERSE_SENSITIVITY):
        cum_regret += best_possible_round_reward - round_reward
        cum_regret_at_round.append(cum_regret)

    plt.plot(rounds_arr[:-1], cum_regret_at_round)

    plt.title('Round vs. Cumulative Regret for VDBE-epsilon')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.legend(loc='upper left')
    plt.show()
