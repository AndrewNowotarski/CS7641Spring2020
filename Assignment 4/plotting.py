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