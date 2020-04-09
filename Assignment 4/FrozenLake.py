# Andrew Nowotarski
# anowotarski3
# CS 7641 ML Spring 2020
# Assignment 4: Markov Decision Processes

# Frozen Lake - Open AI Gym
import numpy as np
import gym
from gym import wrappers
import time
import sys
import Plotting as plt

environment  = 'FrozenLake-v0'
env = gym.make(environment)
env = env.unwrapped
desc = env.unwrapped.desc

time_array=[0]*10
gamma_arr=[0]*10
iters=[0]*10
list_scores=[0]*10

### POLICY ITERATION ####
print('POLICY ITERATION WITH FROZEN LAKE')
for i in range(0,10):
    st=time.time()
    best_policy,k = policy_iteration(env, gamma = (i+0.5)/10)
    scores = evaluate_policy(env, best_policy, gamma = (i+0.5)/10)
    end=time.time()
    gamma_arr[i]=(i+0.5)/10
    list_scores[i]=np.mean(scores)
    iters[i] = k
    time_array[i]=end-st
	
# Plot PI Execution Time. #
plt.plot_pi_execution_time(gamma_arr, time_array, 'ForestManagement')
	# plt.xlabel('Gammas')
	# plt.title('Frozen Lake - Policy Iteration - Execution Time Analysis')
	# plt.ylabel('Execution Time (s)')
	# plt.grid()
	# plt.show()

# Plot PI Rewards. #
plt.plot_pi_rewards(gamma_arr, list_scores, 'ForestManagement')
	# plt.plot(gamma_arr,list_scores)
	# plt.xlabel('Gammas')
	# plt.ylabel('Average Rewards')
	# plt.title('Frozen Lake - Policy Iteration - Reward Analysis')
	# plt.grid()
	# plt.show()

# Plot PI Convergence. #
plt.plot_pi_convergence(gamma_arr, iters, 'ForestManagement')

	# plt.plot(gamma_arr,iters)
	# plt.xlabel('Gammas')
	# plt.ylabel('Iterations to Converge')
	# plt.title('Frozen Lake - Policy Iteration - Convergence Analysis')
	# plt.grid()
	# plt.show()

	
	# ### VALUE ITERATION ###
	# print('VALUE ITERATION WITH FROZEN LAKE')
	# best_vals=[0]*10
	# for i in range(0,10):
	# 	st=time.time()
	# 	best_value,k = value_iteration(env, gamma = (i+0.5)/10)
	# 	policy = extract_policy(env,best_value, gamma = (i+0.5)/10)
	# 	policy_score = evaluate_policy(env, policy, gamma=(i+0.5)/10, n=1000)
	# 	gamma = (i+0.5)/10
	# 	plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (Value Iteration) ' + 'Gamma: '+ str(gamma),policy.reshape(4,4),desc,colors_lake(),directions_lake())
	# 	end=time.time()
	# 	gamma_arr[i]=(i+0.5)/10
	# 	iters[i]=k
	# 	best_vals[i] = best_value
	# 	list_scores[i]=np.mean(policy_score)
	# 	time_array[i]=end-st



	# plt.plot(gamma_arr, time_array)
	# plt.xlabel('Gammas')
	# plt.title('Frozen Lake - Value Iteration - Execution Time Analysis')
	# plt.ylabel('Execution Time (s)')
	# plt.grid()
	# plt.show()

	# plt.plot(gamma_arr,list_scores)
	# plt.xlabel('Gammas')
	# plt.ylabel('Average Rewards')
	# plt.title('Frozen Lake - Value Iteration - Reward Analysis')
	# plt.grid()
	# plt.show()

	# plt.plot(gamma_arr,iters)
	# plt.xlabel('Gammas')
	# plt.ylabel('Iterations to Converge')
	# plt.title('Frozen Lake - Value Iteration - Convergence Analysis')
	# plt.grid()
	# plt.show()

	# plt.plot(gamma_arr,best_vals)
	# plt.xlabel('Gammas')
	# plt.ylabel('Optimal Value')
	# plt.title('Frozen Lake - Value Iteration - Best Value Analysis')
	# plt.grid()
	# plt.show()

	# ### Q-LEARNING #####
	# print('Q LEARNING WITH FROZEN LAKE')
	# st = time.time()
	# reward_array = []
	# iter_array = []
	# size_array = []
	# chunks_array = []
	# averages_array = []
	# time_array = []
	# Q_array = []
	# for epsilon in [0.05,0.15,0.25,0.5,0.75,0.90]:
	# 	Q = np.zeros((env.observation_space.n, env.action_space.n))
	# 	rewards = []
	# 	iters = []
	# 	optimal=[0]*env.observation_space.n
	# 	alpha = 0.85
	# 	gamma = 0.95
	# 	episodes = 30000
	# 	environment  = 'FrozenLake-v0'
	# 	env = gym.make(environment)
	# 	env = env.unwrapped
	# 	desc = env.unwrapped.desc
	# 	for episode in range(episodes):
	# 		state = env.reset()
	# 		done = False
	# 		t_reward = 0
	# 		max_steps = 1000000
	# 		for i in range(max_steps):
	# 			if done:
	# 				break        
	# 			current = state
	# 			if np.random.rand() < (epsilon):
	# 				action = np.argmax(Q[current, :])
	# 			else:
	# 				action = env.action_space.sample()
				
	# 			state, reward, done, info = env.step(action)
	# 			t_reward += reward
	# 			Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
	# 		epsilon=(1-2.71**(-episode/1000))
	# 		rewards.append(t_reward)
	# 		iters.append(i)


	# 	for k in range(env.observation_space.n):
	# 		optimal[k]=np.argmax(Q[k, :])

	# 	reward_array.append(rewards)
	# 	iter_array.append(iters)
	# 	Q_array.append(Q)

	# 	env.close()
	# 	end=time.time()
	# 	#print("time :",end-st)
	# 	time_array.append(end-st)

	# 	# Plot results
	# 	def chunk_list(l, n):
	# 		for i in range(0, len(l), n):
	# 			yield l[i:i + n]

	# 	size = int(episodes / 50)
	# 	chunks = list(chunk_list(rewards, size))
	# 	averages = [sum(chunk) / len(chunk) for chunk in chunks]
	# 	size_array.append(size)
	# 	chunks_array.append(chunks)
	# 	averages_array.append(averages)

	# plt.plot(range(0, len(reward_array[0]), size_array[0]), averages_array[0],label='epsilon=0.05')
	# plt.plot(range(0, len(reward_array[1]), size_array[1]), averages_array[1],label='epsilon=0.15')
	# plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2],label='epsilon=0.25')
	# plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3],label='epsilon=0.50')
	# plt.plot(range(0, len(reward_array[4]), size_array[4]), averages_array[4],label='epsilon=0.75')
	# plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5],label='epsilon=0.95')
	# plt.legend()
	# plt.xlabel('Iterations')
	# plt.grid()
	# plt.title('Frozen Lake - Q Learning - Constant Epsilon')
	# plt.ylabel('Average Reward')
	# plt.show()

	# plt.plot([0.05,0.15,0.25,0.5,0.75,0.95],time_array)
	# plt.xlabel('Epsilon Values')
	# plt.grid()
	# plt.title('Frozen Lake - Q Learning')
	# plt.ylabel('Execution Time (s)')
	# plt.show()

	# plt.subplot(1,6,1)
	# plt.imshow(Q_array[0])
	# plt.title('Epsilon=0.05')

	# plt.subplot(1,6,2)
	# plt.title('Epsilon=0.15')
	# plt.imshow(Q_array[1])

	# plt.subplot(1,6,3)
	# plt.title('Epsilon=0.25')
	# plt.imshow(Q_array[2])

	# plt.subplot(1,6,4)
	# plt.title('Epsilon=0.50')
	# plt.imshow(Q_array[3])

	# plt.subplot(1,6,5)
	# plt.title('Epsilon=0.75')
	# plt.imshow(Q_array[4])

	# plt.subplot(1,6,6)
	# plt.title('Epsilon=0.95')
	# plt.imshow(Q_array[5])
	# plt.colorbar()

	# plt.show()
