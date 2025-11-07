import numpy as np
from matplotlib import pyplot as plt

f = lambda x: 0.1 * np.power(x - 2., 3) - 5. * (x - 2.) + 2.

df = lambda x: 0.3 * np.power(x - 2., 2) - 5.
df_num = lambda f, x, h: (f(x + h) - f(x - h)) * 0.5 / h

tol = 1e-4
maxit = 2000
alpha = 1e-2
x = 12.

plt.ion()

xx = np.linspace(-5, 15, 1001)
yy = f(xx)
plt.plot(xx, yy)

fig = plt.gcf()

plot_ptx, = plt.plot([x], [f(x)], 'r.', markersize=20, linestyle='none')

dfx = df(x)
# dfx = df_num(f, x, 1e-6)
for ii in range(maxit):
  if np.abs(dfx) <= tol: break
  x -= alpha * dfx
  dfx = df(x)
  # dfx = df_num(f, x, 1e-6)
  print(f"\riter = {ii + 1}, x = {x:.2e}, f(x) = {f(x):.2e}, df(x) = {dfx:.2e}", end="")
  plot_ptx.set_data([x], [f(x)])
  plt.draw()
  fig.canvas.flush_events()

print("\n", end="")
plt.ioff()
plt.show()
