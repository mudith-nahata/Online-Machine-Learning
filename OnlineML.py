import numpy as np
#we are using the  numpy model to perform the complex operations easily.
from sklearn import linear_model
#we are using the "scikit learn" module where it consist of the existing Algorithms where it trains the data Incrementally
import time
#time is the predefined module where we can check the learning rate of the trained data and previous data in the Machine Learning Model
n_samples=1
n_features=500
y=np.random.rand(n_samples)
x=np.random.randn(n_samples,n_features)
#here x & y are single data points
clf=linear_model.SGDRegressor()
#Here we are using SGDRegressor where SGD stands for "Stochastic Gradient Descent Regressor" is same as Linear Regression Model
clf.partial_fit(x,y)
#Here partial_fit is used to train the data partially by giving less quantity of data
elapsed_time=time.time()-start_time()
#here elapsed time is used to check how long the single data point is efficiently trained on the Machine Learning Model.
print(elapsed_time)










_
