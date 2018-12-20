import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential, metrics
from keras.layers import Dense, InputLayer
from keras.optimizers import SGD, Adam

np.random.seed(3)
n = 32*50

x = np.linspace(0, 1, n)
func = 0.2 + 0.4*x**2 + 0.3*x*np.sin(15*x) + 0.05*np.cos(50*x)

y = func + np.random.randn(n,)/25

x_train = np.reshape(x ,[n, 1]) 
y_train = np.reshape(y ,[n ,1])

model = Sequential()

model.add(InputLayer(batch_input_shape=(32,1)))
model.add(Dense(32*4, activation='relu'))
model.add(Dense(32*2, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, input_dim=1, activation='linear'))

model.compile(loss='mse', optimizer=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))

model.fit(x_train, y_train, epochs=400)

x_ = np.linspace(0, 1, 160) # define axis

pred_x = np.reshape(x_, [160, 1]) # [160, ] -> [160, 1]
pred_y = model.predict(pred_x, batch_size=160) # predict network output given x_
fig = plt.figure() 
plt.subplot(2,1,1)
plt.plot(x_, 0.2 + 0.4*x_**2 +0.3*x_*np.sin(15*x_) + 0.05*np.cos(50*x_), color = 'g') # plot original function
plt.plot(pred_x, pred_y, 'r') # plot network output
plt.title("Func")
plt.ylabel("Y")
plt.subplot(2,1,2)
plt.plot(x, y, 'ro')
plt.ylabel("Y")
plt.xlabel("X")
plt.show()