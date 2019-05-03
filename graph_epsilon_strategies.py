import matplotlib.pyplot as plt
import epsilon_strategies
import pandas as pd

if __name__ == "__main__":
    stocks = pd.read_csv("stocks/random_stocks.csv")
    tickers = sorted(set([key.split("-")[0] for key in stocks.keys()[1:]]))
    ROUNDS = 2518

    EPSILONS = [0, 0.25, 0.50, 0.75, 1]

    rounds_arr = list(range(ROUNDS))

    # Epsilon-Greedy

    for epsilon in EPSILONS:
        cum_regret = 0
        cum_regret_at_round = []

        for round_reward, best_possible_round_reward in epsilon_strategies.epsilon_greedy(stocks, tickers, ROUNDS, epsilon, ROUNDS, 1):
            cum_regret += best_possible_round_reward - round_reward
            cum_regret_at_round.append(cum_regret)

        plt.plot(rounds_arr, cum_regret_at_round, label='epsilon=' + str(epsilon))

    plt.title('Round vs. Cumulative Regret for Epsilon-Greedy')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.show()

    # Epsilon-First
    for epsilon in EPSILONS:
        cum_regret = 0
        cum_regret_at_round = []

        for round_reward, best_possible_round_reward in epsilon_strategies.epsilon_first(stocks, tickers, ROUNDS, epsilon):
            cum_regret += best_possible_round_reward - round_reward
            cum_regret_at_round.append(cum_regret)

        plt.plot(rounds_arr, cum_regret_at_round, label='epsilon=' + str(epsilon))

    plt.title('Round vs. Cumulative Regret for Epsilon-First')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.legend(loc='upper left')
    plt.show()

    # VDBE-epsilon
    DELTA = 1 / len(tickers)
    INVERSE_SENSITIVITY = 1.5
    cum_regret = 0
    cum_regret_at_round = []

    for round_reward, best_possible_round_reward in epsilon_strategies.VDBE_epsilon_greedy(stocks, tickers, ROUNDS, DELTA, INVERSE_SENSITIVITY):
        cum_regret += best_possible_round_reward - round_reward
        cum_regret_at_round.append(cum_regret)

    plt.plot(rounds_arr[:-1], cum_regret_at_round)

    plt.title('Round vs. Cumulative Regret for VDBE-epsilon')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.legend(loc='upper left')
    plt.show()
