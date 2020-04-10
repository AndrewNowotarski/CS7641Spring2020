# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 4: Markov Decision Processes

# Frozen Grid - Open AI Gym
import numpy as np
import time
import sys
import mdptoolbox
import mdptoolbox.example
import Plotting as plot
import matplotlib.pyplot as plt
import gym

problemName = 'FrozenLakeSmall'
env = gym.make("FrozenLake-v0")

# Convert Open AI Gym problem to work with mdptoolbox. #
nA, nS = env.nA, env.nS
P = np.zeros([nA, nS, nS])
R = np.zeros([nS, nA])
for s in range(nS):
    for a in range(nA):
        transitions = env.P[s][a]
        for p_trans, next_s, reward, _ in transitions:
            P[a,s,next_s] += p_trans
            R[s,a] = reward
        P[a,s,:] /= np.sum(P[a,s,:])

#####################
# Policy Iteration. #
#####################
print('Performing PI Tests...')

value_f = []
policy = []
iterations = []
time_taken = []
epsilons = []

for i in range(0,10):
	pi = mdptoolbox.mdp.PolicyIteration(P, R, (i+0.5)/10)
	pi.run()
	epsilons.append((i+0.5)/10)
	value_f.append(np.mean(pi.V))
	policy.append(pi.policy)
	iterations.append(pi.iter)
	time_taken.append(pi.time)

# Plot Execution Time. #
plot.plot_pi_execution_time(epsilons, time_taken, problemName)

# Plot Rewards. #
plot.plot_pi_rewards(epsilons, value_f, problemName)

# Plot Convergence. #
plot.plot_pi_convergence(epsilons, iterations, problemName)

####################
# Value Iteration. #
####################
print('Performing VI Tests...')

value_f = []
policy = []
iterations = []
time_taken = []
epsilons = []

for i in range(0,10):
	pi = mdptoolbox.mdp.ValueIteration(P, R, (i+0.5)/10)
	pi.run()
	epsilons.append((i+0.5)/10)
	value_f.append(np.mean(pi.V))
	policy.append(pi.policy)
	iterations.append(pi.iter)
	time_taken.append(pi.time)

# Plot Execution Time. #
plot.plot_vi_execution_time(epsilons, time_taken, problemName)

# Plot Rewards. #
plot.plot_vi_rewards(epsilons, value_f, problemName)

# Plot Convergence. #	
plot.plot_vi_convergence(epsilons, iterations, problemName)
	
###############
# Q-Learning. #
###############

##################
# Tune Discount. #
##################
print('Performing Q-Learning Tests - Discount Rate')

value_f = []
policy = []
iterations = []
time_taken = []
epsilons = []

for discountRate in [0.05,0.15,0.25,0.5,0.75,0.95]:
	pi = mdptoolbox.mdp.QLearning(P, R, discountRate, n_iter=1000000)
	pi.run()
	epsilons.append(discountRate)
	iterations.append(1000000)
	value_f.append(np.mean(pi.V))
	policy.append(pi.policy)
	time_taken.append(pi.time)

# Plot Convergence. #
plot.plot_qlearning_convergence(epsilons, iterations, problemName, '-DiscountRate')

# Plot Execution Time. #
plot.plot_qlearning_execution_time(epsilons, time_taken, problemName, '-DiscountRate')

# Plot Rewards. #
plot.plot_qlearning_rewards(epsilons, value_f, problemName, '-DiscountRate')

####################
# Tune Iterations. #
####################
print('Performing Q-Learning Tests - Iterations')

value_f = []
policy = []
iterations = []
time_taken = []
epsilons = []

for iteration in [10000, 15000, 20000, 25000, 50000, 75000, 100000, 250000, 500000, 1000000]:
	pi = mdptoolbox.mdp.QLearning(P, R, .95, n_iter=iteration)
	pi.run()
	epsilons.append(iteration)
	iterations.append(iteration)
	value_f.append(np.mean(pi.V))
	policy.append(pi.policy)
	time_taken.append(pi.time)

# Plot Convergence. #
plot.plot_qlearning_convergence(epsilons, iterations, problemName, '-Iterations')

# Plot Execution Time. #
plot.plot_qlearning_execution_time(epsilons, time_taken, problemName, '-Iterations')

# Plot Rewards. #
plot.plot_qlearning_rewards(epsilons, value_f, problemName, '-Iterations')