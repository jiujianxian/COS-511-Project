import random
import pandas as pd
import numpy as np
import math

def calc_rewards_arms(bernoulli_params):
    arms_rewards = []
    for p in bernoulli_params:
        rand = np.random.random()
        reward = 1 if rand < p else 0
        arms_rewards.append(reward)

    return arms_rewards

def epsilon_greedy(bernoulli_params, num_rounds, epsilon, rounds_until_decrease, decreasing_factor):
    num_arms = len(bernoulli_params)
    times_arm_selected = np.zeros((num_arms,))
    arms_avg_reward = np.zeros((num_arms,))

    for t in range(num_rounds):

        if (t+1) % rounds_until_decrease == 0:
            epsilon *= decreasing_factor

        rand = random.uniform(0, 1)

        # Calculate the reward for all the arms
        arms_rewards = calc_rewards_arms(bernoulli_params)
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


def epsilon_first(bernoulli_params, num_rounds, epsilon):
    num_explorative_rounds = math.ceil(epsilon * num_rounds)
    num_arms = len(bernoulli_params)
    times_arm_selected = np.zeros((num_arms,))
    arms_avg_reward = np.zeros((num_arms,))

    for t in range(num_rounds):
        rand = random.uniform(0, 1)

        # Calculate the reward for all the arms
        arms_rewards = calc_rewards_arms(bernoulli_params)
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


def VDBE_epsilon_greedy(bernoulli_params, num_rounds, delta, inverse_sensitivity):
    epsilon = 1
    num_arms = len(bernoulli_params)
    times_arm_selected = np.zeros((num_arms,))
    arms_avg_reward = np.zeros((num_arms,))

    # Calculate the reward for all the arms
    arms_rewards = calc_rewards_arms(bernoulli_params)

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
        next_arms_rewards = calc_rewards_arms(bernoulli_params)

        temporal_diff_err = next_arms_rewards[chosen_arm_idx] - round_reward
        power = - abs(step_size * temporal_diff_err) / inverse_sensitivity
        boltzmann_val = (1 - math.exp(power)) / (1 + math.exp(power))

        epsilon = delta * boltzmann_val + (1-delta)*epsilon

        arms_rewards = next_arms_rewards

        yield round_reward, best_possible_round_reward


if __name__ == "__main__":
    EPSILON = 0.3
    ROUNDS = 10000

    bernoulli_params = np.linspace(0.1, 0.9, 20)

    print("------ Epsilon Greedy --------")
    round = 1
    cum_regret = 0
    cumulative_rewards = []
    best_possible_cumulative_rewards = []
    for round_reward, best_possible_round_reward in epsilon_greedy(bernoulli_params, ROUNDS, EPSILON, ROUNDS, 1):
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
    for round_reward, best_possible_round_reward in epsilon_first(bernoulli_params, ROUNDS, EPSILON):
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
    for round_reward, best_possible_round_reward in epsilon_greedy(bernoulli_params, ROUNDS, EPSILON, ROUND_PER_DECREASE, DECREASE_FACTOR):
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
    DELTA = 1 / len(bernoulli_params)
    INVERSE_SENSITIVITY = 1.5
    round = 1
    cum_regret = 0
    cumulative_rewards = []
    best_possible_cumulative_rewards = []
    for round_reward, best_possible_round_reward in VDBE_epsilon_greedy(bernoulli_params, ROUNDS, DELTA, INVERSE_SENSITIVITY):
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