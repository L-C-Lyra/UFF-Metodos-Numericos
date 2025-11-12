import numpy as np
from matplotlib import pyplot as plt

f = lambda x1, x2: 0.1 * np.power(x1 - 2., 3) - 5. * (x1 - 2.) + 0.1 * np.power(x2 - 2., 3) - 5. * (x2 - 2.) + 2.

df1 = lambda x1, x2: 0.3 * np.power(x1 - 2., 2) - 5.
df1_num = lambda f, x1, x2, h: (f(x1 + h, x2) - f(x1 - h, x2)) * 0.5 / h

df2 = lambda x1, x2: 0.3 * np.power(x2 - 2., 2) - 5.
df2_num = lambda f, x1, x2, h: (f(x1, x2 + h) - f(x1, x2 - h)) * 0.5 / h

tol = 1e-4
maxit = 2000
alpha = 1e-2
x = (12., 12.)

plt.ion()

xx = np.linspace(-5, 15, 1001)
fgrid = f(*np.meshgrid(xx, xx)).reshape(1001, 1001)
plt.imshow(fgrid, cmap='CMRmap')
plt.colorbar()

fig = plt.gcf()

def coords2pixels(x, d, img):
  x_min = d[0]
  x_max = d[1]
  px = np.zeros_like(x)
  
  px[0] = (x[0] - x_min) / (x_max - x_min) * img.shape[0]
  px[1] = (x[1] - x_min) / (x_max - x_min) * img.shape[1]

  return px


px = coords2pixels(x, (-5., 15.), fgrid)
plot_ptx, = plt.plot([px[0]], [px[1]], '.', color='lime', markersize=20, linestyle='none')

dfx = np.array([df1(x[0], x[1]), df2(x[0], x[1])])
for ii in range(maxit):
  if np.linalg.norm(dfx) <= tol: break
  x -= alpha * dfx
  dfx = np.array([df1(x[0], x[1]), df2(x[0], x[1])])
  print(f"\riter = {ii + 1}, x = [{x[0]:.2e}, {x[1]:.2e}], f(x, y) = {f(x[0], x[1]):.2e}, df(x, y) = [{dfx[0]:.2e}, {dfx[1]:.2e}]", end="")
  px = coords2pixels(x, (-5., 15.), fgrid)
  plot_ptx.set_data([px[0]], [px[1]])
  plt.draw()
  fig.canvas.flush_events()

print("\n", end="")
plt.ioff()
plt.show()
