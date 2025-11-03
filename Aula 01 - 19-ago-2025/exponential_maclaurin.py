import numpy as np
from matplotlib import pyplot as plt

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1)


dx = 1.
xmax = 5.
xmin = -5.
n = int((xmax - xmin) / dx)
x = np.linspace(xmin, xmax, n, dtype='float32')
y = np.exp(x, dtype='float32')
ax1.plot(x, y, 'b.')

_ = input(" ")

dx = 0.2
n = int((xmax - xmin) / dx)
x = np.linspace(xmin, xmax, n, dtype='float32')
y = np.exp(x, dtype='float32')
ax1.plot(x, y, 'r+')


_ = input(" ")


dx = 0.05
n = int((xmax - xmin) / dx)
x = np.linspace(xmin, xmax, n, dtype='float32')
y = np.zeros(x.shape, dtype='float32')
err = np.zeros(x.shape, dtype='float32')
y_exp = np.exp(x, dtype='float32')

for ii in range(10):
    y[:] += np.power(x, ii, dtype='float32') * ( 1 / np.math.factorial(ii))
    err[:] = np.abs(y_exp - y)
    ax1.plot(x, y)
    ax2.plot(x, err)
    print(ii)

    _ = input(" ")
