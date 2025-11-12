import numpy as np

def simpson_13_rule(f, a, b, n_c):
  n = 2 * n_c
  h = (b - a) / n
  res = f(a) + f(b)

  x = np.linspace(a, b - h, n)
  res += 4. * np.sum(f(x[1::2])) # 1, 3, 5, ...
  res += 2. * np.sum(f(x[2::2])) # 2, 4, 6, ...

  return res * h / 3.


def simpson_38_rule(f, a, b, n_c):
  n = 3 * n_c
  h = (b - a) / n
  res = f(a) + f(b)

  x = np.linspace(a, b - h, n)
  res += 3. * np.sum(f(x[1::3])) # 1, 4, 7, ...
  res += 3. * np.sum(f(x[2::3])) # 2, 5, 8, ...
  res += 2. * np.sum(f(x[3::3])) # 3, 6, 9, ...

  return res * h * 3. / 8.


if __name__ == "__main__":
  f = lambda x: 1 / x
  a, b = 1., 3.
  exact_sol = np.log(b) - np.log(a)
  print(f"Resultado Esperado: {exact_sol:.8f}\n")

  print("1/3 de Simpson:")
  for n in range(2, 17, 2):
    q = simpson_13_rule(f, a, b, n)
    err = np.abs((q - exact_sol) / exact_sol)
    print(f"{n:2d} Curvas: {q:.8f}, Erro Relativo: {err:.2e}")

  print("\n3/8 de Simpson:")
  for n in range(2, 17, 2):
    q = simpson_38_rule(f, a, b, n)
    err = np.abs((q - exact_sol) / exact_sol)
    print(f"{n:2d} Curvas: {q:.8f}, Erro Relativo: {err:.2e}")
