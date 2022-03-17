pi = 3.1415
e = 2.1783


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


class t:
    def __init__(self, degrees_freedom):
        self.degrees_freedom = degrees_freedom

    def pdf(self, x):
        nu = self.degrees_freedom
        return gamma((nu + 1) / 2) / ((pi * nu) ** (1 / 2) * gamma(nu / 2)) * (1 + (x ** 2) / nu) ** (-(nu + 1) / 2)

    def test(self, test_statistic, right=True, steps=1000):
        area = 0
        sign = 1
        if not right:
            sign = -1
        for step in range(steps):
            x = test_statistic + sign * step
            area += self.pdf(x)
        return area


class Z:
    def __init__(self):
        pass

    def pdf(self, x):
        mu = 0
        sd = 1
        return 1 / (2 * (pi * sd)) ** (1/2) * e ** (-(x - mu) / sd)

    def test(self, test_statistic, right=True, steps=1000):
        area = 0
        sign = 1
        if not right:
            sign = -1
        for step in range(steps):
            x = test_statistic + sign * step
            area += self.pdf(x)
        return area


class Chi_squared:
    def __init__(self, degrees_freedom):
        self.degrees_freedom = degrees_freedom

    def pdf(self, x):
        return (Z.pdf(x) ** 2) * self.degrees_freedom

    def test(self, test_statistic, steps=1000):
        area = 0
        for step in range(steps):
            x = test_statistic + step
            area += self.pdf(x)
        return area


class F:
    def __init__(self, degrees_freedom1, degrees_freedom2):
        self.degrees_freedom1 = degrees_freedom1
        self.degrees_freedom2 = degrees_freedom2
        self.Chi_squared1 = Chi_squared(degrees_freedom1)
        self.Chi_squared2 = Chi_squared(degrees_freedom2)

    def pdf(self, x):
        return (self.Chi_squared1.pdf(x) / self.degrees_freedom1) / (self.Chi_squared2.pdf(x) / self.degrees_freedom2)

    def test(self, test_statistic, steps=1000):
        area = 0
        for step in range(steps):
            x = test_statistic + step
            area += self.pdf(x)
        return area
