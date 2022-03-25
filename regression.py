from matrix import Matrix
from hypothesis_testing import t, Z, ChiSquared, F
from math import log

# Options to functions
class Regression:
    def __init__(self, data, header=True, partition=None, restriction=False, intercept=True):
        self.partition = partition

        self.y = []

        if not partition:
            self.X = []
            self.XT = None
            self.XTX_inverse = None

            self.beta_hat_ols = None

            self.var_hat = None

        if restriction:
            self.J = None
            self.R = None
            self.RT = None
            self.c = None
            self.XTX_inverseRT = None
            self.RXTX_inverseRT_inverse = None
            self.D = None
            self.XD = None
            self.XDT = None

            self.e_hatTe_hat = None
            self.dTRXTX_inverseRT_inversed = None
            self.beta_star = None
            self.test_stat = None

            self.F = None
            self.p_value = None

        if partition:
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
        self.s_squared = None

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
                    if intercept:
                        self.X.append([1])
                    else:
                        self.X.append([])
                else:
                    if intercept:
                        self.X1.append([1])
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
            self.s_squared = (self.e_hat.transpose().multiply(self.e_hat)).scale(self.X.nrows - self.X.ncols).data[0][0]
            self.var_hat = self.XTX_inverse.scale(self.s_squared)
        else:
            raise ValueError("Please supply data which has more rows than columns, otherwise inference will not be possible.")
        return {"Beta_hat_OLS": self.beta_hat_ols}

    def restricted_ols(self, R, c, F_test = False):
        self.J = len(R)
        if len(R[0]) != self.X.ncols or self.J != len(c):
            raise ValueError("Please supply a valid restriction matrix and constant vector.")
        if not self.beta_hat_ols:
            self.ols()
        self.R = Matrix(R)
        self.RT = self.R.transpose()
        self.c = Matrix(c)
        self.XTX_inverseRT = self.XTX_inverse.multiply(self.RT)
        self.RXTX_inverseRT_inverse = self.R.multiply(self.XTX_inverseRT).inverse()
        self.D = self.XTX_inverseRT.multiply(self.RXTX_inverseRT_inverse.multiply((self.R.multiply(self.beta_hat_ols)).subtract(self.c)))
        self.XD = self.X.multiply(self.D)
        self.XDT = self.XD.transpose()
        self.e_hatTe_hat = self.s_squared * (self.X.nrows - self.X.ncols)
        self.dTRXTX_inverseRT_inversed = self.e_hatTe_hat + (self.XDT.multiply(self.XD)).data[0][0]
        self.beta_star = self.beta_hat_ols.subtract(self.D)
        if F_test:
            self.test_stat = self.dTRXTX_inverseRT_inversed * ((self.X.nrows - self.X.ncols) / (self.J * self.e_hatTe_hat))
            self.F = F(self.J, self.X.nrows - self.beta_hat_ols.nrows)
            self.p_value = self.F.test(self.test_stat)
        else:
            self.test_stat = self.dTRXTX_inverseRT_inversed / self.J
            self.Chi_2 = ChiSquared(self.J)
            self.p_value = self.Chi_2.test(self.test_stat)
        return {"Beta_star": self.beta_star, "p-value": self.p_value}

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
        if self.X1.nrows > (self.X1.ncols + self.X2.ncols):
            self.s_squared = (self.e_hat.transpose().multiply(self.e_hat)).scale(self.X1.nrows - self.X1.ncols - self.X2.ncols).data[0][0]
            self.var_hat_1 = self.X1TM2X1_inverse.scale(self.s_squared)
            self.var_hat_2 = self.X2TM1X2_inverse.scale(self.s_squared)
        else:
            raise ValueError("Please supply data which has more rows than columns, otherwise inference will not be possible.")
        return {"Beta_hat_FWL_1": self.beta_hat_fwl_1, "Beta_hat_FWL_2": self.beta_hat_fwl_2}

    def wald_heteroskedasticity_test(self, exponential=False):
        data = self.e_hat.data
        c = []
        for i in range(len(data)):
            data[i][0] *= data[i][0]
            if exponential:
                data[i][0] = log(data[i][0])
            data[i] += self.X.data[i]
        R = Matrix.identity_matrix(self.ncols - 1).data
        for i in range(len(R)):
            c.append([0])
            R[i].insert(0, 0)
        wald = Regression(data, restriction=True, intercept=False)
        result = wald.restricted_ols(R, c)
        return {"Beta_hat_WALD": wald.beta_hat_ols.data, "p-value": result["p-value"]}

    def breusch_pagan_heteroskedasticity_test(self):
        pass

    def white_heteroskedasticity_test(self):
        pass

    def t_test(self, beta_hat, var_hat, degrees_of_freedom, mu_0=0):
        if not (self.beta_hat_ols or self.beta_hat_fwl_1 or self.beta_hat_fwl_2):
            self.ols()
        p_values = []
        students_t = t(degrees_of_freedom)
        var_hat_data = var_hat.data
        for i in range(len(beta_hat)):
            p_values.append(students_t.test((beta_hat[i] - mu_0) / var_hat_data[i][i] ** (1 / 2)) * 2)
        return {"p-values": p_values}

    def z_test(self, beta_hat, var_hat, mu_0=0):
        if not (self.beta_hat_ols or self.beta_hat_fwl_1 or self.beta_hat_fwl_2):
            self.ols()
        p_values = []
        standard_normal = Z()
        var_hat_data = var_hat.data
        for i in range(len(beta_hat)):
            p_values.append(standard_normal.test((beta_hat[i] - mu_0) / var_hat_data[i][i] ** (1 / 2)) * 2)
        return {"p-values": p_values}
