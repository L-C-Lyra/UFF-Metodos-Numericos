import numpy as np
from matplotlib import pyplot as plt

plt.ion()
fig1, (ax1, ax2) = plt.subplots(2, 1)
fig2, (ax3, ax4) = plt.subplots(2, 1)

fig1.suptitle("Aproximação com Erro de Ordem Linear")
fig2.suptitle("Aproximação com Erro de Ordem Quadrática")

dx = 0.1
xmax = 2. * np.pi
xmin = 0.
n = int((xmax - xmin) / dx)
x = np.linspace(xmin, xmax, n, dtype='float32')
y_cos = np.cos(x, dtype='float32')

ax1.plot(x, y_cos)
ax2.plot(x, np.abs(y_cos - y_cos))

ax3.plot(x, y_cos)
ax4.plot(x, np.abs(y_cos - y_cos))

_ = input(" ")

for dx in [np.pi * 0.2, 
           np.pi * 0.1, 
           np.pi * 0.05, 
           np.pi * 0.025, 
           np.pi * 0.0125, 
           np.pi * 0.00625, 
           np.pi * 1e-05, 
           np.pi * 1e-06,
           np.pi * 1e-07]:
    n = int((xmax - xmin) / dx)
    x = np.linspace(xmin, xmax, n, dtype='float32')
    y_sin = np.sin(x, dtype='float32')
    y_cos = np.cos(x, dtype='float32')
    y_der = (y_sin[1:] - y_sin[:-1]) / dx

    ax1.plot(x[:-1], y_der)
    ax2.plot(x[:-1], np.abs(y_der - y_cos[:-1]))

    y_der_center = (y_sin[2:] - y_sin[:-2]) / (2 * dx)

    ax3.plot(x[1:-1], y_der_center)
    ax4.plot(x[1:-1], np.abs(y_der_center - y_cos[1:-1]))

    _ = input(f"dx = {dx:.2e}")
