class Matrix:
    def __init__(self, data):
        self.nrows = len(data)
        self.ncols = len(data[0])
        for i in range(1, self.nrows):
            if len(data[i]) != self.ncols:
                raise ValueError("The lengths of the rows of the data you submitted are not equal.")
        self.data = data

    def dimensions(self):
        return [self.nrows, self.ncols]

    def transpose(self):
        data = []
        for i in range(self.ncols):
            data.append([])
        for i in range(self.ncols):
            for j in range(self.nrows):
                data[i].append(self.data[j][i])
        return Matrix(data)

    def system(self):
        if self.nrows != self.ncols:
            raise ValueError("Number of rows has to be equal to the number of columns to invert a matrix.")
        data = self.data
        for i in range(self.nrows):
            for j in range(self.nrows):
                if i == j:
                    data[i].append(1)
                else:
                    data[i].append(0)
        return Matrix(data)

    @staticmethod
    def row_echelon(row, index):
        factor = row[index]
        length = len(row)
        for i in range(length):
            row[i] = row[i] / factor
        return row

    @staticmethod
    def reduce(reducor, reduced, index):
        if reduced[index] == 0:
            return reduced
        factor = reduced[index] / reducor[index]
        length = len(reduced)
        for i in range(length):
            reduced[i] = reduced[i] - factor * reducor[i]
        return reduced

    @staticmethod
    def dotproduct(a, b):
        if len(a) != len(b):
            raise ValueError("The lengths of your vectors must be the same.")
        length = len(a)
        sum = 0
        for i in range(length):
            sum += a[i] * b[i]
        return sum

    def add_sub(self, B, add=True):
        if self.ncols != B.ncols or self.nrows != B.nrows:
            raise ValueError("The number of rows and columns for both matrices must be the same.")
        result = []
        for i in range(self.nrows):
            result.append([])
            for j in range(self.ncols):
                if add:
                    result[i].append(self.data[i][j] + B.data[i][j])
                else:
                    result[i].append(self.data[i][j] - B.data[i][j])
        return Matrix(result)

    def add(self, B):
        return self.add_sub(B)

    def subtract(self, B):
        return self.add_sub(B, False)

    def scale(self, alfa):
        result = []
        for i in range(self.nrows):
            result.append([])
            for j in range(self.ncols):
                result[i].append(alfa * self.data[i][j])
        return Matrix(result)

    def multiply(self, B):
        if self.ncols != B.nrows:
            raise ValueError("The columns of your left matrix are not equal to those of your right matrix.")
        A = self.data
        B_T = B.transpose().data
        result = []
        for i in range(self.nrows):
            result.append([])
            for j in range(B.ncols):
                result[i].append(self.dotproduct(A[i], B_T[j]))
        return Matrix(result)

    def invert(self):
        system = self.system()
        data = system.data
        for j in range(self.nrows):
            for i in range(j, self.nrows):
                if data[i][j] != 0:
                    data[i] = self.row_echelon(data[i], j)
                    if i != j:
                        data[j], data[i] = data[i], data[j]
                    for k in range(self.nrows):
                        if k != j:
                            data[k] = self.reduce(data[j], data[k], j)
                    break
                if i == self.nrows - 1:
                    raise ValueError("You have a supplied a non-invertible matrix (perfect multicollinearity).")
        inverse = []
        for i in range(self.nrows):
            inverse.append([])
            for j in range(self.nrows):
                inverse[i].append(data[i][self.nrows + j])
        return Matrix(inverse)
