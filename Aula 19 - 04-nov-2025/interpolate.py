import numpy as np
from matplotlib import pyplot as plt
from spline import LinearSplineInterpolator as LSI
from spline import CubicSplineInterpolator as CSI

N = 20

x = np.linspace(0, N - 1, N, dtype='float32')
y = (np.random.rand(N) * 2. - 1.) * np.max(x)

x += np.random.rand(N) * 0.2 - 0.1

l_interpolator = LSI(x, y)
l_interpolator.fit()

c_interpolator = CSI(x, y)
c_interpolator.fit()

xx = np.linspace(np.min(x), np.max(x), 10 * N + 1, dtype=x.dtype)
lyy = l_interpolator.predict(xx)
cyy = c_interpolator.predict(xx)

plt.plot(xx, lyy, 'g')
plt.plot(xx, cyy, 'b--')

plt.plot(x, y, 'r.', linestyle='None', markersize=12)
plt.legend(['Spline Linear', 'Spline Cúbica', 'Dataset'])
plt.show()
