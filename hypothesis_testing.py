pi = 3.14159265358979
e = 2.71828182845904


def factorial(n):
    if n < 2:
        return 1
    else:
        return n * factorial(n - 1)


def gamma(z):
    if z <= 0:
        raise ValueError("The gamma function is only defined for strictly positive numbers.")
    if z < 1:
        return 1/z
    n = int(z)
    lower = factorial(n - 1)
    upper = lower * n
    return lower + (upper - lower) * (z - n) ** 2


def beta(alfa, beta):
    return (gamma(alfa) * gamma(beta)) / gamma(alfa + beta)


class t:
    def __init__(self, degrees_freedom):
        self.degrees_freedom = degrees_freedom

    def pdf(self, x):
        nu = self.degrees_freedom
        return gamma((nu + 1) / 2) / ((pi * nu) ** (1 / 2) * gamma(nu / 2)) * (1 + (x ** 2) / nu) ** (-1 * (nu + 1) / 2)

    def test(self, test_statistic, right=True, steps=1000):
        area = 0
        sign = 1
        step_size = 0.01
        if not right:
            sign = -1
        for step in range(steps):
            x = test_statistic + sign * (step + 0.5) * step_size
            area += self.pdf(x) * step_size
        return area


class Z:
    def __init__(self):
        pass

    def pdf(self, x):
        return (e ** ((-1 * (x ** 2)) / 2)) / ((2 * pi) ** (1/2))

    def test(self, test_statistic, right=True, steps=1000):
        area = 0
        sign = 1
        step_size = 0.01
        if not right:
            sign = -1
        for step in range(steps):
            x = test_statistic + sign * (step + 0.5) * step_size
            area += self.pdf(x) * step_size
        return area


class ChiSquared:
    def __init__(self, degrees_freedom):
        self.degrees_freedom = degrees_freedom

    def pdf(self, x):
        k = self.degrees_freedom
        return ((x ** (k / 2 - 1)) * (e ** ((-1 * x) / 2))) / ((2 ** (k / 2)) * gamma(k / 2))

    def test(self, test_statistic, steps=1000):
        area = 0
        step_size = 0.01
        for step in range(steps):
            x = test_statistic + (step + 0.5) * step_size
            area += self.pdf(x) * step_size
        return area


class F:
    def __init__(self, degrees_freedom1, degrees_freedom2):
        self.degrees_freedom1 = degrees_freedom1
        self.degrees_freedom2 = degrees_freedom2

    def pdf(self, x):
        d1 = self.degrees_freedom1
        d2 = self.degrees_freedom2
        return ((((d1 * x) ** d1 * d2 ** d2) / ((d1 * x + d2) ** (d1 + d2))) ** (1 / 2)) / (x * beta(d1 / 2, d2 / 2))

    def test(self, test_statistic, steps=1000):
        area = 0
        step_size = 0.01
        for step in range(steps):
            x = test_statistic + (step + 0.5) * step_size
            area += self.pdf(x) * step_size
        return area
