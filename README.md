# Machine-Learning Projects 

This repository contains a collection of projects which focus on solving rudimentary Multi class classification, Natural language processing and Vision problems. 
The projects explore the application of classical Machine Learning algorithms to solve the problems. The implementations primarily use only the Numpy and Pandas libraries.
The implementation of the framework from scratch provides a great environment to experiment and learn from the code. The following algorithms have been used in the projects:

1. Multi Nomial Naive Bayes Classifier 
2. Logistic Regression
3. K Nearest Neighbour
4. Decision Trees
5. Multi Layer Perceptron 

The repository uses publicly available datesets which have been extensively studied in literature. The datasets used are given below:
1. Wisconsin Breast Cancer dataset (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
2. Hepatitis (http://archive.ics.uci.edu/ml/datasets/Hepatitis)
3. Twenty News Group Dataset
4. IMDB Reviews Dataset(http://ai.stanford.edu/ ̃amaas/data/sentiment/)
5. MNIST Dataset

The projects explore the effect of changing the hyperparameters and the architecture of the models on the performance. The repository is intended to be a playground for students 
who would like to change different aspects the models to understand the finer nuances. The author has tried to write the code in a clear, lucid and modular manner to 
enable effficient implementation of the code. However, there is a scope to further streamline and improve the code. The authors have run a limited set of experiments to 
to understand the effect of changing the hyperparameters on the model performance. Further tuning the hyperparameters is expected to improve the peformance of the models.

All the codes can be run on the CPU and provide satisfactory performance. The Multi Layer perceptron code provides the option of runnning the code on GPU. The code uses 
CuPY which is the GPU version of Numpy. Also, the natural language processing code requires a large amount of memory to store the word tokens. The primary focus of the projects
is to improve the understanding of the models and thus, the code does not provide a comparison with literature.
