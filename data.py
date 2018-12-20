import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)
n = 32*10

x = np.linspace(0, 1, n)
func = 0.2 + 0.4*x**2 + 0.3*x*np.sin(15*x) + 0.05*np.cos(50*x)

y = func + np.random.randn(n,)/75

fig = plt.figure() 

plt.plot(x, y, 'ro')
plt.ylabel("Y")
plt.xlabel("X")
plt.show()