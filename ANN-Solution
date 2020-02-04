#The average root mean square error (RMSE) of the model is 0.299
#Importing the important modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import tensorflow.keras as keras

from keras.models import Sequential
from keras.layers import Dense
data1={}

#Loading the data
dataset = pd.read_csv('https://raw.githubusercontent.com/bxs-machine-learning-club/Iris-Setosa/master/Iris.csv')

#Drop Id
dataset=dataset.drop(axis=1,columns=['Id'])

#Replace Species with Numbers
dataset=dataset.replace('Iris-setosa',1)
dataset=dataset.replace('Iris-versicolor',2)
dataset=dataset.replace('Iris-virginica',3)

#Seperate the data
data1['data']=dataset.drop(axis=1,columns=['Species']).values.tolist()
data1['target']=dataset['Species'].values.tolist()

#Scaling all the features so that they are in between 0 and 1
x_scaled = scale(data1['data'])
y_scaled = scale(data1['target'])

#Dividing the data between training and testing
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled)

#Neural Network:
#Sequential: A linearly connected layers

#Activation: Defines the output of that node given an input or set of inputs. ReLU is a popular one

#Input layer: Takes in the 4 features of each point in the datset and passes it though an activation method.

#Hidden Layer: Outputs 4 points, after passing through an activation method

#Output Layer: Outputs one point (the guess), after using the previous layer's points and passing it through an activation method. This time, it's linear because we are using linear regression
model = Sequential([
   Dense(11, activation="relu", input_dim=4),
   Dense(4, activation="relu"),
   Dense(1, activation="linear")
])

#Compiler:
#Loss: sets our metric of error to MSE

#Optimizer: Tries to decrease the metric of error over time. ADAM is a popular open

#Metrics: The "scores" that will be given after the learning is finished
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error", "mean_absolute_error"])

#Fitting:
#Epochs: The amount of times the neural network trains.

#Validation Split: An extra testing set based off the training set for the neural network to compare results with.
model.fit(x_train, y_train, epochs=1000, validation_split=0.2)


#Evaluating how good our model was
loss, mse, mae = model.evaluate(x_test, y_test)
print('The average root mean square error (RMSE) of the model is {:5.3f}'.format(np.sqrt(mse)))
