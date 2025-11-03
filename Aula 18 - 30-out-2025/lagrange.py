import numpy as np

class LagrangeInterpolator:
  __x: np.ndarray
  __y: np.ndarray
  num_points: int
  weights: np.ndarray | None


  def __init__(self, x: np.ndarray, y: np.ndarray):
    self.__x = np.array(x)
    self.__y = np.array(y)
    self.num_points = self.__x.size
    self.weights = None

    if self.__y.size != self.num_points: raise ValueError(f"Size Mismatch: x ({self.__x.size}), y ({self.__y.size})")
  

  def fit(self):
    n = self.num_points
    mask = np.ones(n, dtype='bool')
    self.weights = np.zeros(n, dtype=self.__x.dtype)

    for ii in range(n):
      mask[ii] = False
      self.weights[ii] = np.prod(1. / (self.__x[ii] - self.__x[mask]))
      mask[ii] = True
  

  def __predict_lazy(self, cx: np.ndarray):
    n = self.num_points
    mask = np.ones(n, dtype='bool')
    L = np.zeros((cx.size, n), dtype=self.__x.dtype)
    x_diff = cx[:, None] - self.__x # [x - x1, x - x2, ..., x - xn]

    for ii in range(n):
      mask[ii] = False
      L[:, ii] = np.prod(x_diff[:, mask] / (self.__x[ii] - self.__x[mask]), axis=1)
      mask[ii] = True
    
    return L @ self.__y
  

  def predict(self, cx: np.ndarray | float | int):
    if isinstance(cx, (int, float)): cx = np.array([cx], dtype=self.__x.dtype)
    if self.weights is None: return self.__predict_lazy(cx)

    n = self.num_points
    mask = np.ones(n, dtype='bool')
    L = np.zeros((cx.size, n), dtype=self.__x.dtype)
    x_diff = cx[:, None] - self.__x # [x - x1, x - x2, ..., x - xn]

    for ii in range(n):
      mask[ii] = False
      L[:, ii] = np.prod(x_diff[:, mask], axis=1) * self.weights[ii]
      mask[ii] = True
    
    return L @ self.__y
