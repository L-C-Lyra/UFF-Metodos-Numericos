import numpy as np

class VandermondeInterpolator:
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
    A = np.logspace(0, n - 1, n, base=self.__x).T
    self.weights = np.linalg.solve(A, self.__y)
  

  def predict(self, cx: np.ndarray | float):
    if self.weights is None: raise ValueError(f"predict() requires interpolation weights. fit() must be called first. ")

    n = self.num_points
    mtx = np.logspace(0, n - 1, n, base=cx).T

    return np.dot(mtx, self.weights)
