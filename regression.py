from matrix import Matrix


# Options to functions
class Regression:
    def __init__(self, data, header=True, partition=None):
        self.partition = partition

        self.y = []

        if not partition:
            self.X = []
            self.XT = None
            self.XTX_inverse = None

            self.beta_hat_ols = None

            self.var_hat = None
        else:
            self.X1 = []
            self.X2 = []
            self.X1T = None
            self.X2T = None
            self.M1 = None
            self.M2 = None
            self.X1TM2 = None
            self.X2TM1 = None
            self.X1TM2X1_inverse = None
            self.X2TM1X2_inverse = None

            self.beta_hat_fwl_1 = None
            self.beta_hat_fwl_2 = None

            self.var_hat_1 = None
            self.var_hat_2 = None

        self.e_hat = None
        self.sigma_squared_hat = None

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
                if not partition:
                    self.X.append([])
                else:
                    self.X1.append([])
                    self.X2.append([])
                for j in range(self.ncols):
                    if j == 0:
                        self.y[i - 1].append(float(data[i][j]))
                    else:
                        if not partition:
                            self.X[i - 1].append(float(data[i][j]))
                        else:
                            if j < partition:
                                self.X1[i - 1].append(float(data[i][j]))
                            else:
                                self.X2[i - 1].append(float(data[i][j]))

        self.y = Matrix(self.y)

        if not partition:
            self.X = Matrix(self.X)
        else:
            self.X1 = Matrix(self.X1)
            self.X2 = Matrix(self.X2)

    def ols(self):
        if self.partition:
            return self.partitioned_ols()
        self.XT = self.X.transpose()
        self.XTX_inverse = (self.XT.multiply(self.X)).inverse()
        self.beta_hat_ols = self.XTX_inverse.multiply(self.XT).multiply(self.y)
        self.e_hat = self.y.subtract(self.X.multiply(self.beta_hat_ols))
        if self.X.nrows > self.X.ncols:
            self.sigma_squared_hat = (self.e_hat.transpose().multiply(self.e_hat)).scale(self.X.nrows - self.X.ncols).data[0][0]
            self.var_hat = self.XTX_inverse.scale(self.sigma_squared_hat)
        else:
            raise ValueError("Please supply data which has more rows than columns, otherwise inference will not be possible.")
        return self.beta_hat_ols

    def partitioned_ols(self):
        if not self.partition:
            return self.ols()
        self.X1T = self.X1.transpose()
        self.X2T = self.X2.transpose()
        self.M1 = Matrix.identity_matrix(self.X1.nrows).subtract(self.X1.multiply((self.X1T.multiply(self.X1)).inverse()).multiply(self.X1T))
        self.M2 = Matrix.identity_matrix(self.X2.nrows).subtract(self.X2.multiply((self.X2T.multiply(self.X2)).inverse()).multiply(self.X2T))
        self.X1TM2 = self.X1T.multiply(self.M2)
        self.X2TM1 = self.X2T.multiply(self.M1)
        self.X1TM2X1_inverse = (self.X1TM2.multiply(self.X1)).inverse()
        self.X2TM1X2_inverse = (self.X2TM1.multiply(self.X2)).inverse()
        self.beta_hat_fwl_1 = self.X1TM2X1_inverse.multiply(self.X1TM2).multiply(self.y)
        self.beta_hat_fwl_2 = self.X2TM1X2_inverse.multiply(self.X2TM1).multiply(self.y)
        self.e_hat = self.y.subtract(self.X1.multiply(self.beta_hat_fwl_1)).subtract(self.X2.multiply(self.beta_hat_fwl_2))
        if self.X1.nrows > (self.X1.ncols + self.X2.nrows):
            self.sigma_squared_hat = (self.e_hat.transpose().multiply(self.e_hat)).scale(self.X1.nrows - self.X1.ncols - self.X2.ncols).data[0][0]
            self.var_hat_1 = self.X1TM2X1_inverse.scale(self.sigma_squared_hat)
            self.var_hat_2 = self.X2TM1X2_inverse.scale(self.sigma_squared_hat)
        else:
            raise ValueError("Please supply data which has more rows than columns, otherwise inference will not be possible.")
        return [self.beta_hat_fwl_1, self.beta_hat_fwl_2]
