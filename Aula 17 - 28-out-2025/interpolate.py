import numpy as np
from matplotlib import pyplot as plt
from newton import NewtonInterpolator as NI

N = 5

x = np.linspace(0, N - 1, N, dtype='float32')
y = (np.random.rand(N) * 2. - 1.) * np.max(x)

x += np.random.rand(N) * 0.2 - 0.1
y += np.random.rand(N) * np.mean(y) * 0.1 - 0.05

interpolator = NI(x, y)
interpolator.fit()

xx = np.linspace(np.min(x), np.max(x), 1001, dtype='float32')
yy = interpolator.predict(xx)

plt.plot(xx, yy, 'b--')

plt.plot(x, y, 'r.', linestyle='None', markersize=12)
plt.legend(['Newton', 'Dataset'])
plt.show()
