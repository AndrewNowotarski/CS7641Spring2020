# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 4: Markov Decision Processes

# Forest Management - MDPToolbox
import numpy as np
import time
import sys
import mdptoolbox
import mdptoolbox.example
import Plotting as plot
import matplotlib.pyplot as plt

problemName = 'ForestManagement-Small'
P, R = mdptoolbox.example.forest(S=500)

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
print('Performing Q-Learning Tests...')

# value_f = []
# policy = []
# iters = []
# time_array = []
# Q_table = []
# rew_array = []

# for discountRate in [0.05,0.15,0.25,0.5,0.75,0.95]:
# 	pi = mdptoolbox.mdp.QLearning(P, R, discountRate)
# 	pi.run()
# 	# rew_array.append(pi.)
# 	value_f.append(np.mean(pi.V))
# 	policy.append(pi.policy)
# 	iters.append(pi.iter)
# 	time_array.append(pi.time)
# 	Q_table.append(pi.Q)
	
# # Plot Execution Time. #
# plot.plot_qlearning_execution_time(gamma_arr, time_array, problemName)

# # Plot Rewards. #
# plot.plot_qlearning_rewards(gamma_arr, value_f, problemName)

# # Plot Convergence. #	
# plot.plot_qlearning_convergence(gamma_arr, iters, problemName)

# 	plt.plot(range(0,10000), rew_array[0],label='epsilon=0.05')
# 	plt.plot(range(0,10000), rew_array[1],label='epsilon=0.15')
# 	plt.plot(range(0,10000), rew_array[2],label='epsilon=0.25')
# 	plt.plot(range(0,10000), rew_array[3],label='epsilon=0.50')
# 	plt.plot(range(0,10000), rew_array[4],label='epsilon=0.75')
# 	plt.plot(range(0,10000), rew_array[5],label='epsilon=0.95')
# 	plt.legend()
# 	plt.xlabel('Iterations')
# 	plt.grid()
# 	plt.title('Forest Management - Q Learning - Decaying Epsilon')
# 	plt.ylabel('Average Reward')
# 	plt.show()

# 	plt.subplot(1,6,1)
# 	plt.imshow(Q_table[0][:20,:])
# 	plt.title('Epsilon=0.05')

# 	plt.subplot(1,6,2)
# 	plt.title('Epsilon=0.15')
# 	plt.imshow(Q_table[1][:20,:])

# 	plt.subplot(1,6,3)
# 	plt.title('Epsilon=0.25')
# 	plt.imshow(Q_table[2][:20,:])

# 	plt.subplot(1,6,4)
# 	plt.title('Epsilon=0.50')
# 	plt.imshow(Q_table[3][:20,:])

# 	plt.subplot(1,6,5)
# 	plt.title('Epsilon=0.75')
# 	plt.imshow(Q_table[4][:20,:])

# 	plt.subplot(1,6,6)
# 	plt.title('Epsilon=0.95')
# 	plt.imshow(Q_table[5][:20,:])
# 	plt.colorbar()
# 	plt.show()

# 	return