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

def plot_qlearning_execution_time(xValues, yValues, problemName, tuningType):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Time (seconds)', problemName + ' Q-Learning Execution Time' + tuningType, 'Charts/' + problemName + '/QLearning_execution_time' + tuningType + '.png')

def plot_vi_rewards(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Average Rewards', problemName + ' VI Rewards', 'Charts/' + problemName + '/VI_rewards.png')

def plot_pi_rewards(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Average Rewards', problemName + ' PI Rewards', 'Charts/' + problemName + '/PI_rewards.png')

def plot_qlearning_rewards(xValues, yValues, problemName, tuningType):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Average Rewards', problemName + ' Q-Learning Rewards' + tuningType, 'Charts/' + problemName + '/QLearning_rewards' + tuningType + '.png')

def plot_vi_convergence(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Iterations', problemName + ' VI Convergence', 'Charts/' + problemName + '/VI_convergence.png')

def plot_pi_convergence(xValues, yValues, problemName):
    plot_generic_line(xValues, 'Epsilon', yValues, 'Iterations', problemName + ' PI Convergence', 'Charts/' + problemName + '/PI_convergence.png')

def plot_qlearning_convergence(xValues, yValues, problemName, tuningType):
    plot_generic_line(xValues, 'Discount Rate', yValues, 'Iterations', problemName + ' Q-Learning Convergence' + tuningType, 'Charts/' + problemName + '/QLearning_convergence' + tuningType + '.png')

def plot_qlearning_decaying_epsilon(rewards, q, problemName):

    plt.plot(range(0,10000), rew_array[0],label='epsilon=0.05')
    plt.plot(range(0,10000), rew_array[1],label='epsilon=0.15')
    plt.plot(range(0,10000), rew_array[2],label='epsilon=0.25')
    plt.plot(range(0,10000), rew_array[3],label='epsilon=0.50')
    plt.plot(range(0,10000), rew_array[4],label='epsilon=0.75')
    plt.plot(range(0,10000), rew_array[5],label='epsilon=0.95')
    plt.legend()
    plt.xlabel('Iterations')
    plt.grid()
    plt.title(problemName)
    plt.ylabel('Average Reward')

    plt.subplot(1,6,1)
    plt.imshow(Q_table[0][:20,:])
    plt.title('Epsilon=0.05')

    plt.subplot(1,6,2)
    plt.title('Epsilon=0.15')
    plt.imshow(Q_table[1][:20,:])

    plt.subplot(1,6,3)
    plt.title('Epsilon=0.25')
    plt.imshow(Q_table[2][:20,:])

    plt.subplot(1,6,4)
    plt.title('Epsilon=0.50')
    plt.imshow(Q_table[3][:20,:])

    plt.subplot(1,6,5)
    plt.title('Epsilon=0.75')
    plt.imshow(Q_table[4][:20,:])

    plt.subplot(1,6,6)
    plt.title('Epsilon=0.95')
    plt.imshow(Q_table[5][:20,:])
    plt.colorbar()

    plt.savefig('Charts/' + problemName + '/QLearning_decaying_epsilon.png')
    plt.clf()