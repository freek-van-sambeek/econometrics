from matrix import Matrix


# Options to functions
class Regression:
    def __init__(self, data, header=True):
        self.y = []
        self.X = []
        self.e_hat = None
        self.sigma_squared_hat = None
        self.var_hat = None
        self.beta_hat_ols = None
        self.labels = []
        self.nrows = len(data)
        self.ncols = len(data[0])
        for i in range(self.nrows):
            if header:
                for j in range(self.ncols):
                    self.labels.append(data[0][j])
                header = False
            else:
                self.y.append([])
                self.X.append([])
                for j in range(self.ncols):
                    if j == 0:
                        self.y[i - 1].append(float(data[i][j]))
                    else:
                        self.X[i - 1].append(float(data[i][j]))
        self.y = Matrix(self.y)
        self.X = Matrix(self.X)

    def ols(self):
        self.beta_hat_ols = (self.X.transpose().multiply(self.X)).invert().multiply(self.X.transpose()).multiply(self.y)
        self.e_hat = self.y.subtract(self.X.multiply(self.beta_hat_ols))
        if self.X.nrows > self.X.ncols:
            self.sigma_squared_hat = (self.e_hat.transpose().multiply(self.e_hat)).scale(self.X.nrows - self.X.ncols)
        self.var_hat = ((self.X.transpose().multiply(self.X)).invert()).scale(self.sigma_squared_hat.data[0][0])
        return self.beta_hat_ols
