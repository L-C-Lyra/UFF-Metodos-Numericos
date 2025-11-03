import numpy as np

class NewtonInterpolator:
  __x: np.ndarray
  __y: np.ndarray
  num_points: int
  weights: np.ndarray | None


  def __init__(self, x: np.ndarray, y: np.ndarray):
    self.__x = np.array(x)
    self.__y = np.array(y)
    self.num_points = self.__x.size
    self.A = None
    self.weights = None

    if self.__y.size != self.num_points: raise ValueError(f"Size Mismatch: x ({self.__x.size}), y ({self.__y.size})")
  

  def fit(self):
    n = self.num_points
    table = np.zeros((n, n), dtype=self.__y.dtype)
    table[0, :] = self.__y

    for i in range(1, n):
      table[i, :n - i] = (table[i - 1, 1:n - i + 1] - table[i - 1, :n - i]) / (self.__x[i:] - self.__x[:n - i])

    self.weights = np.array(table[:, 0])
  

  def predict(self, cx: np.ndarray | float):
    if self.weights is None: raise ValueError(f"predict() requires interpolation weights. fit() must be called first. ")
    if isinstance(cx, (int, float)): cx = np.array([cx], dtype=self.__x.dtype)

    n = self.num_points
    x_diff = cx[:, None] - self.__x # [x - x1, x - x2, ..., x - xn]
    x_arr = np.ones((cx.size, n), dtype=cx.dtype)

    for i in range(1, n):
      x_arr[:, i] = np.prod(x_diff[:, :i], axis=1)
    
    return np.dot(x_arr, self.weights)
