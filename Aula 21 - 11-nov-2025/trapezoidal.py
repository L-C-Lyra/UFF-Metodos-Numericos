import numpy as np

def trapezoidal_rule(f, a, b, n):
  h = (b - a) / n
  res = (f(a) + f(b)) * 0.5

  if n < 2: return res * h

  x = np.linspace(a + h, b - h, n - 1.)
  res += np.sum(f(x))

  return res * h


if __name__ == "__main__":
  f = lambda x: 1 / x
  a, b = 1., 3.
  exact_sol = np.log(b) - np.log(a)
  print(f"Resultado Esperado: {exact_sol:.8f}\n")

  for n in range(2, 17, 2):
    q = trapezoidal_rule(f, a, b, n)
    err = np.abs((q - exact_sol) / exact_sol)
    print(f"{n:2d} Trapézios: {q:.8f}, Erro Relativo: {err:.2e}")
