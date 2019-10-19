import numpy as np

class LinearRegression():

    def __init__(self):
        self.coef = None
        self.bias = None
        self.x = None
        self.y = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.learning_rate = 0.01
        self.epoch = 10000

    def make_data(self, data_number):
        '''

        :return: Data for linear equation y = -3 + 2x1 + 3x2 + 4x3 + 5x4
        '''
        np.random.seed(0)
        self.x = np.random.randint(10, size=(data_number, 4))
        self.y = [(-3 + 2*x[0] + 3*x[1] + 4*x[2] + 5*x[3]) for x in self.x]
        train_number = int(len(self.x) * 0.7)
        self.train_x = self.x[:train_number]
        self.train_y = self.y[:train_number]
        self.test_x = self.x[train_number:]
        self.test_y = self.y[:train_number]

    @staticmethod
    def mse(x1, x2):
        return np.mean(np.square(np.subtract(x1, x2)))

    def fit(self):
        coef = np.zeros(4)
        bias = 0

        for each in range(self.epoch):
            y_pred = [(np.dot(x, coef) + bias) for x in self.train_x]
            residual = np.subtract(y_pred, self.train_y)

            # gradient descent of bias part
            bias_gradient = np.mean(residual)
            bias = bias - (self.learning_rate * bias_gradient)

            # gradient descent of coefficient part
            coef_gradients = [self.learning_rate * np.mean(np.multiply(self.train_x[:, i], residual)) for i in range(len(coef))]
            coef = np.subtract(coef, coef_gradients)

            print("Epoch-{} has been end".format(each+1))
            print("Epoch-{0} loss is {1}".format(each+1, self.mse(y_pred, self.train_y)))

        self.coef = coef
        self.bias = bias


if __name__ == '__main__':
    lr = LinearRegression()
    lr.make_data(100)
    lr.fit()
    print(lr.coef)
    print(lr.bias)
