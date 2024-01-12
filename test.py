def f(x):
    return x**2

def use_f(c, x):
    return c(x)


print(use_f(f, 2))