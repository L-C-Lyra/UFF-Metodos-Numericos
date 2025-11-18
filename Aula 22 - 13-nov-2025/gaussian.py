import numpy as np

def gaussian_quadrature(f, a, b, n):
  y, w = np.polynomial.legendre.leggauss(n)

  t = lambda x: 0.5 * (b + a + (b - a) * x)

  res = 0.
  for ii in range(n):
    res += w[ii] * f(t(y[ii]))
  res *= 0.5 * (b - a)

  return res


if __name__ == "__main__":
  f = lambda x: 1 + (x - 1) ** 3

  q = gaussian_quadrature(f, 1., 2., 2)

  print(f"Resultado: {q}")
