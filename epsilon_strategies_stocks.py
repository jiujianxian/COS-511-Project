import random
import pandas as pd
import numpy as np
import math



def single_day_reward(stocktable, name, day, amountToInvest=1.0):
    if name == "brk":
        name = "brk-a"
    openPrice = stocktable[name + "-open"][day]
    closePrice = stocktable[name + "-close"][day]
    sharesBought = amountToInvest / openPrice
    amountAfterSale = sharesBought * closePrice

    return amountAfterSale - amountToInvest

def calc_daily_stock_changes(stocks, tickers, day, amountToInvest=1.0):
    stock_changes = []

    for ticker in tickers:
        open_price = stocks["{0}-open".format(ticker)][day]
        close_price = stocks["{0}-close".format(ticker)][day]
        shares_bought = amountToInvest / open_price
        amount_after_sale = shares_bought * close_price
        net_return = amount_after_sale - amountToInvest

        #net_return = single_day_reward(stocks, ticker, day, amountToInvest)

        stock_changes.append(net_return)

    return stock_changes


def epsilon_greedy(stocks, tickers, num_rounds, epsilon, rounds_until_decrease, decreasing_factor):
    num_arms = len(tickers)
    times_arm_selected = np.zeros((num_arms,))
    arms_avg_reward = np.zeros((num_arms,))

    for t in range(num_rounds):

        if (t+1) % rounds_until_decrease == 0:
            epsilon *= decreasing_factor

        rand = random.uniform(0, 1)

        # Calculate the reward for all the arms
        arms_rewards = calc_daily_stock_changes(stocks, tickers, t)
        # Get possible reward in round
        best_possible_round_reward = max(arms_rewards)

        chosen_arm_idx = -1
        if rand > epsilon and t > 0:      # Exploitation face
            chosen_arm_idx = np.argmax(arms_avg_reward)
        else:       # Exploration face
            # Choose a random arm
            chosen_arm_idx = random.randint(0, num_arms - 1)

        round_reward = arms_rewards[chosen_arm_idx]

        times_arm_selected[chosen_arm_idx] += 1
        arms_avg_reward[chosen_arm_idx] += (1.0 / times_arm_selected[chosen_arm_idx]) * (round_reward - arms_avg_reward[chosen_arm_idx])

        yield round_reward, best_possible_round_reward


def epsilon_first(stocks, tickers, num_rounds, epsilon):
    num_explorative_rounds = math.ceil(epsilon * num_rounds)
    num_arms = len(tickers)
    times_arm_selected = np.zeros((num_arms,))
    arms_avg_reward = np.zeros((num_arms,))

    for t in range(num_rounds):
        rand = random.uniform(0, 1)

        # Calculate the reward for all the arms
        arms_rewards = calc_daily_stock_changes(stocks, tickers, t)
        # Get possible reward in round
        best_possible_round_reward = max(arms_rewards)

        chosen_arm_idx = -1
        if t > num_explorative_rounds:  # Exploitation face
            chosen_arm_idx = np.argmax(arms_avg_reward)
        else:  # Exploration face
            # Choose a random arm
            chosen_arm_idx = random.randint(0, num_arms - 1)

        round_reward = arms_rewards[chosen_arm_idx]

        times_arm_selected[chosen_arm_idx] += 1
        arms_avg_reward[chosen_arm_idx] += (1.0 / times_arm_selected[chosen_arm_idx]) * (
                    round_reward - arms_avg_reward[chosen_arm_idx])

        yield round_reward, best_possible_round_reward


def VDBE_epsilon_greedy(stocks, tickers, num_rounds, delta, inverse_sensitivity):
    epsilon = 1
    num_arms = len(tickers)
    times_arm_selected = np.zeros((num_arms,))
    arms_avg_reward = np.zeros((num_arms,))

    # Calculate the reward for all the arms
    arms_rewards = calc_daily_stock_changes(stocks, tickers, 0)

    for t in range(num_rounds-1):
        rand = random.uniform(0, 1)

        # Get possible reward in round
        best_possible_round_reward = max(arms_rewards)

        chosen_arm_idx = -1
        if rand > epsilon and t > 0:      # Exploitation face
            chosen_arm_idx = np.argmax(arms_avg_reward)
        else:       # Exploration face
            # Choose a random arm
            chosen_arm_idx = random.randint(0, num_arms - 1)

        round_reward = arms_rewards[chosen_arm_idx]

        times_arm_selected[chosen_arm_idx] += 1
        step_size = (1.0 / times_arm_selected[chosen_arm_idx])
        arms_avg_reward[chosen_arm_idx] += step_size * (round_reward - arms_avg_reward[chosen_arm_idx])

        # Calculate the reward for all the arms
        next_arms_rewards = calc_daily_stock_changes(stocks, tickers, t+1)

        temporal_diff_err = next_arms_rewards[chosen_arm_idx] - round_reward
        power = - abs(step_size * temporal_diff_err) / inverse_sensitivity
        boltzmann_val = (1 - math.exp(power)) / (1 + math.exp(power))

        epsilon = delta * boltzmann_val + (1-delta)*epsilon

        arms_rewards = next_arms_rewards

        yield round_reward, best_possible_round_reward


if __name__ == "__main__":
    stocks = pd.read_csv("stocks/random_stocks.csv")
    tickers = sorted(set([key.split("-")[0] for key in stocks.keys()[1:]]))

    EPSILON = 0.3
    ROUNDS = 2518

    print("------ Epsilon Greedy --------")
    round = 1
    cum_regret = 0
    cumulative_rewards = []
    best_possible_cumulative_rewards = []
    for round_reward, best_possible_round_reward in epsilon_greedy(stocks, tickers, ROUNDS, EPSILON, ROUNDS, 1):
        cumulative_rewards.append(round_reward)
        best_possible_cumulative_rewards.append(best_possible_round_reward)

        cum_regret += best_possible_round_reward - round_reward
        #print("{:3d}:::Round reward: {:.3f} ---- Best possible round reward: {:.3f} ----- Cum regret: {:.3f}".format(round, round_reward, best_possible_round_reward, cum_regret))
        round += 1

    print("Total Return: " + str(np.sum(cumulative_rewards)))
    print("Expected Payoff: " + str(np.mean(cumulative_rewards)))
    print("Payoff Variance: " + str(np.var(cumulative_rewards)))

    print("Best Total Return: " + str(np.sum(best_possible_cumulative_rewards)))
    print("Best Possible Expected Payoff: " + str(np.mean(best_possible_cumulative_rewards)))
    print("Best Possible Payoff Variance: " + str(np.var(best_possible_cumulative_rewards)))

    print("Cumulative regret: " + str(cum_regret))

    print()
    print()

    print("------ Epsilon First --------")
    round = 1
    cum_regret = 0
    cumulative_rewards = []
    best_possible_cumulative_rewards = []
    for round_reward, best_possible_round_reward in epsilon_first(stocks, tickers, ROUNDS, EPSILON):
        cumulative_rewards.append(round_reward)
        best_possible_cumulative_rewards.append(best_possible_round_reward)

        cum_regret += best_possible_round_reward - round_reward
        #print("{:3d}:::Round reward: {:.3f} ---- Best possible round reward: {:.3f} ----- Cum regret: {:.3f}".format(round, round_reward, best_possible_round_reward, cum_regret))
        round += 1

    print("Total Return: " + str(np.sum(cumulative_rewards)))
    print("Expected Payoff: " + str(np.mean(cumulative_rewards)))
    print("Payoff Variance: " + str(np.var(cumulative_rewards)))

    print("Best Total Return: " + str(np.sum(best_possible_cumulative_rewards)))
    print("Best Possible Expected Payoff: " + str(np.mean(best_possible_cumulative_rewards)))
    print("Best Possible Payoff Variance: " + str(np.var(best_possible_cumulative_rewards)))

    print("Cumulative regret: " + str(cum_regret))

    print()
    print()

    print("------ Epsilon Decreasing --------")
    ROUND_PER_DECREASE = int(ROUNDS / 10)
    DECREASE_FACTOR = 0.9
    round = 1
    cum_regret = 0
    cumulative_rewards = []
    best_possible_cumulative_rewards = []
    for round_reward, best_possible_round_reward in epsilon_greedy(stocks, tickers, ROUNDS, EPSILON, ROUND_PER_DECREASE, DECREASE_FACTOR):
        cumulative_rewards.append(round_reward)
        best_possible_cumulative_rewards.append(best_possible_round_reward)

        cum_regret += best_possible_round_reward - round_reward
        # print("{:3d}:::Round reward: {:.3f} ---- Best possible round reward: {:.3f} ----- Cum regret: {:.3f}".format(round, round_reward, best_possible_round_reward, cum_regret))
        round += 1

    print("Total Return: " + str(np.sum(cumulative_rewards)))
    print("Expected Payoff: " + str(np.mean(cumulative_rewards)))
    print("Payoff Variance: " + str(np.var(cumulative_rewards)))

    print("Best Total Return: " + str(np.sum(best_possible_cumulative_rewards)))
    print("Best Possible Expected Payoff: " + str(np.mean(best_possible_cumulative_rewards)))
    print("Best Possible Payoff Variance: " + str(np.var(best_possible_cumulative_rewards)))

    print("Cumulative regret: " + str(cum_regret))

    print()
    print()

    print("------ VDBE Epsilon Greedy --------")
    DELTA = 1 / len(tickers)
    INVERSE_SENSITIVITY = 1.5
    round = 1
    cum_regret = 0
    cumulative_rewards = []
    best_possible_cumulative_rewards = []
    for round_reward, best_possible_round_reward in VDBE_epsilon_greedy(stocks, tickers, ROUNDS, DELTA, INVERSE_SENSITIVITY):
        cumulative_rewards.append(round_reward)
        best_possible_cumulative_rewards.append(best_possible_round_reward)

        cum_regret += best_possible_round_reward - round_reward
        #print("{:3d}:::Round reward: {:.3f} ---- Best possible round reward: {:.3f} ----- Cum regret: {:.3f}".format(round, round_reward, best_possible_round_reward, cum_regret))
        round += 1

    print("Total Return: " + str(np.sum(cumulative_rewards)))
    print("Expected Payoff: " + str(np.mean(cumulative_rewards)))
    print("Payoff Variance: " + str(np.var(cumulative_rewards)))

    print("Best Total Return: " + str(np.sum(best_possible_cumulative_rewards)))
    print("Best Possible Expected Payoff: " + str(np.mean(best_possible_cumulative_rewards)))
    print("Best Possible Payoff Variance: " + str(np.var(best_possible_cumulative_rewards)))

    print("Cumulative regret: " + str(cum_regret))

    print()
    print()