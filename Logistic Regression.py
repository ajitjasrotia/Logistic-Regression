#Logistic Regression Algorithm

import numpy as np

#Load Training and Testing Data Set
train = data.data
test = data.target

#Training Data for Logistic Regression
X_train = train[]
Y_train = train[]
 

#Testing Data for Logistic Regression 
X_test = test[]
Y_test = test[]


#Parameters Setup
rate = 0.001
iterations = 1000
weights = np.zeros((X_train.shape[1],1))
gradient = np.zeros((X_train.shape[1],1))

#Function to Calcualte Hypothesis
def sigmoid(data):
    return 1 / (1+np.exp(-data))

#Function to Minimize theta
def minimize(th,grad):
    th = th - (rate * grad)
    return th

#Function for Likelihood 

#Iterations for Gradient Descent
for i in range(iterations):
    predict = X_train.dot(theta)
    prediction = sigmoid(predict)
    
