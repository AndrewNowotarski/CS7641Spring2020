CS7641 Spring 2020 - Assignment 1 README
Andrew Nowotarski
anowotarski3

Instructions:

1.	All of the code is available at my public Github repo located at 
	https://github.com/AndrewNowotarski/CS7641Spring2020/tree/master/Assignment%201
	
2.	All of the generated charts used in the analysis document are contained in the Charts directory. Each dataset has a sub-directory of rounds of 
	tuning and results for both validation and learning curve data. 
	
3.	There are two experiments you can run to reproduce the charts and my results; runexperiment1.py will run the analysis on the Titantic dataset 
	while runexperiment2.py runs the analysis for the Pima Indian Diabetes dataset.
	
4.	Each experiement has multiple 'rounds' of analysis where I manually experimented with different hyper-parameter values to see what the optimum was
	in terms of lowest variance / lowest bias. Each round builds on the next as I chose the best HP value and hard coded that for the next round for 
	each model. You can uncomment the code for each model for each round to see how the performance changes over time with the model complexity / 
	learning curve charts.
	
5.	I used a random seed value on the dataset split to ensure that the results are reproducable.

6.	Both datasets are contained in the Git repo. They are untouched beyond manually updating the diabetes column in diabetes.csv to use boolean values
	rather than classes. I did this cleaning manually in Excel on the column. The titanic_survivor.csv dataset came pre-processed from Kaggle and is untouched.
	
7.	Both datasets are available through Kaggle:
	Titanic:
		https://www.kaggle.com/ak1352/titanic-cl
	Pima Indian Diabetes
		https://www.kaggle.com/omnamahshivai/dataset-pima-indians-binary-classification
		
8.	Have fun! I know I did.