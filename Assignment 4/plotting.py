# Andrew Nowotarski
# anowotarski
# CS 7641 ML Spring 2020
# Assignment 4: Markov Decision Process
import matplotlib.pyplot as plt

def plot_generic_line(xValues, xLabel, yValues, yLabel, title, filename):
    plt.plot(xValues, yValues)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()

def plot_vi_execution_time(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Time (seconds)', problemName + ' VI Execution Time', 'Charts/' + problemName + '/VI_execution_time.png')

def plot_pi_execution_time(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Time (seconds)', problemName + ' PI Execution Time', 'Charts/' + problemName + '/PI_execution_time.png')

def plot_qlearning_execution_time(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Time (seconds)', problemName + ' Q-Learning Execution Time', 'Charts/' + problemName + '/QLearning_execution_time.png')

def plot_vi_rewards(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Average Rewards', problemName + ' VI Rewards', 'Charts/' + problemName + '/VI_rewards.png')

def plot_pi_rewards(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Average Rewards', problemName + ' PI Rewards', 'Charts/' + problemName + '/PI_rewards.png')

def plot_qlearning_rewards(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Average Rewards', problemName + ' Q-Learning Rewards', 'Charts/' + problemName + '/QLearning_rewards.png')

def plot_vi_convergence(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Iterations', problemName + ' VI Convergence', 'Charts/' + problemName + '/VI_convergence.png')

def plot_pi_convergence(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Iterations', problemName + ' PI Convergence', 'Charts/' + problemName + '/PI_convergence.png')

def plot_qlearning_convergence(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Discount Rate', yValues, 'Iterations', problemName + ' Q-Learning Convergence', 'Charts/' + problemName + '/QLearning_convergence.png')


def plot_policy_map(title, policy, map_desc, color_map, direction_map):
	fig = plt.figure()
	ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
	font_size = 'x-large'
	if policy.shape[1] > 16:
		font_size = 'small'
	plt.title(title)
	for i in range(policy.shape[0]):
		for j in range(policy.shape[1]):
			y = policy.shape[0] - i - 1
			x = j
			p = plt.Rectangle([x, y], 1, 1)
			p.set_facecolor(color_map[map_desc[i,j]])
			ax.add_patch(p)

			text = ax.text(x+0.5, y+0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
						   horizontalalignment='center', verticalalignment='center', color='w')
			

	plt.axis('off')
	plt.xlim((0, policy.shape[1]))
	plt.ylim((0, policy.shape[0]))
	plt.tight_layout()
	plt.savefig(title+str('.png'))
	plt.close()

	return (plt)