import numpy as np
import scipy.linalg
from abc import ABC, abstractmethod

class SplineInterpolator(ABC):
  _x: np.ndarray
  _y: np.ndarray
  num_points: int
  weights: np.ndarray | None

  
  def __init__(self, x: np.ndarray, y: np.ndarray):
    indexes = np.argsort(x)
    self._x = np.array(x[indexes])
    self._y = np.array(y[indexes])
    self.num_points = self._x.size
    self.weights = None

    if self._y.size != self.num_points: raise ValueError(f"Size Mismatch: x ({self._x.size}), y ({self._y.size})")
  

  def array_search(self, x_new, is_sorted=False):
    mask = np.zeros(x_new.size, dtype='bool')
    indexes = np.zeros(x_new.size, dtype='uint32')

    if not is_sorted:
      sorted_indexes = np.argsort(x_new)
      x_sorted = np.array(x_new[sorted_indexes])
    else:
      x_sorted = x_new
    
    mask[:] = x_sorted < self._x[0]
    start = np.sum(mask)
    mask[:start] = False

    for ii in range(1, self.num_points - 1):
      mask[start:] = x_sorted[start:] < self._x[ii]
      sum_true = np.sum(mask)

      if sum_true > 0.: indexes[start:start + sum_true] = ii - 1.

      mask[start:start + sum_true] = False
      start += sum_true

      if start >= x_sorted.size: break
    
    if start < x_sorted.size: indexes[start:] = self.num_points - 2.
    if is_sorted: return indexes

    indexes_unsorted = np.zeros_like(indexes)
    indexes_unsorted[sorted_indexes] = indexes[:]

    return indexes_unsorted


  def binary_search(self, x_new):
    if x_new < self._x[0]: return 0.
    if x_new > self._x[-1]: return self.num_points - 2.

    begin = 0.
    end = self.num_points - 1.
    
    ii = (self.num_points - 1.) // 2.
    while True:
      if self._x[ii] <= x_new and self._x[ii + 1] >= x_new: return ii
      if x_new > self._x[ii]: begin = ii
      else: end = ii

      ii = (begin + end) // 2.
  

  @abstractmethod
  def fit(self):
    pass

  @abstractmethod
  def predict(self, cx: np.ndarray | float | int):
    pass


class LinearSplineInterpolator(SplineInterpolator):
  def __init__(self, x: np.ndarray, y: np.ndarray):
    super().__init__(x, y)
  

  def fit(self):
    dx = self._x[1:] - self._x[:-1]
    dy = self._y[1:] - self._y[:-1]

    self.weights = dy / dx
  

  def predict(self, cx: np.ndarray | float | int, is_sorted=False):
    if isinstance(cx, (int, float)): ii = self.binary_search(cx)
    else: ii = self.array_search(cx, is_sorted=is_sorted)

    dx = cx - self._x[ii]

    return self._y[ii] + self.weights[ii] * dx


class CubicSplineInterpolator(SplineInterpolator):
  def __init__(self, x: np.ndarray, y: np.ndarray):
    super().__init__(x, y)


  def fit(self):
    h = self._x[1:] - self._x[:-1]
    diags = np.zeros((3, self.num_points))

    # Upper Diagonal
    diags[0, 2:] = h[1:]
    # Main Diagonal
    diags[1, 0] = 1.
    diags[1, 1:-1] = 2. * (h[1:] + h[:-1])
    diags[1, -1] = 1.
    # Lower Diagonal
    diags[2, :-2] = h[:-1]

    f = np.zeros_like(self._y)
    dy = self._y[1:] - self._y[:-1]
    f[1:-1] = 3. * (dy[1:] / h[1:] - dy[:-1] / h[:-1])

    c = scipy.linalg.solve_banded((1, 1), diags, f) # (1, 1) means one Upper and one Lower Diagonal

    self.weights = np.zeros((4, self.num_points - 1))
    self.weights[0, :] = np.array(self._y[:-1])                         # a_i
    self.weights[1, :] = dy / h - (1. / 3.) * h * (c[1:] + 2. * c[:-1]) # b_i
    self.weights[2, :] = c[:-1]                                         # c_i
    self.weights[3, :] = (1. / 3.) * (c[1:] - c[:-1]) / h               # d_i
  

  def predict(self, cx: np.ndarray | float | int, is_sorted=False):
    if isinstance(cx, (int, float)): ii = self.binary_search(cx)
    else: ii = self.array_search(cx, is_sorted=is_sorted)

    dx = cx - self._x[ii]
    a = self.weights[0, ii]
    b = self.weights[1, ii]
    c = self.weights[2, ii]
    d = self.weights[3, ii]

    return a + b * dx + c * dx**2 + d * dx**3
