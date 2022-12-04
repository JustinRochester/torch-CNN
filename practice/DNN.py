import numpy as np
import math
import matplotlib.pyplot as plt

input_size, output_size = 2, 1


def dis(pred, labels):
    loss_value = np.sum(np.square(pred - labels), axis=1)
    return np.sqrt(np.average(loss_value, axis=1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    x_new = sigmoid(x)
    return x_new * (1 - x_new)


def leaky_relu(x):
    x_new = np.copy(x)
    x_new[x_new < 0] *= 0.1
    return x_new


def d_leaky_relu(x):
    x_new = np.copy(x)
    x_new[x_new > 0] = 1
    x_new[x_new <= 0] = 0.1
    return x_new


def relu(x):
    x_new = np.copy(x)
    x_new[x_new < 0] = 0
    return x_new


def d_relu(x):
    x_new = np.copy(x)
    x_new[x_new > 0] = 1
    x_new[x_new <= 0] = 0
    return x_new


activation_functions = {
    "sigmoid": (sigmoid, d_sigmoid),
    "relu": (relu, d_relu),
    "leaky_relu": (leaky_relu, d_leaky_relu)
}


class NeuralUnits:
    def __init__(self, shape=(1, 1), alpha=0.0):
        self.val = np.random.random_sample(size=shape)
        self.grad = np.zeros(shape)
        self.first_moment = np.zeros(shape)
        self.second_moment = np.zeros(1)
        self.t = 0
        self.alpha = alpha


class NeuralLayer:
    def __init__(self, pre_size=1, size=1,
                 activation_name='relu',
                 alpha=0.0,
                 beta1=0.9, beta2=0.999):
        self.pre_output = NeuralUnits((pre_size, 1), alpha)
        self.w = NeuralUnits((size, pre_size), alpha)
        self.bias = NeuralUnits((size, 1), alpha)
        self.input = NeuralUnits((size, 1), alpha)

        activation = activation_functions[activation_name]
        self.forward_function = activation[0]
        self.backward_function = activation[1]
        self.beta1 = beta1
        self.beta2 = beta2
        self.update_list = [self.w, self.bias]

    def regular_loss(self):
        loss_val = 0
        for elem in self.update_list:
            loss_val += np.sum(np.square(elem.val)) * elem.alpha
        return loss_val / 2

    def forward(self, pre_output, dropout_probability=1):
        self.pre_output.val = np.average(pre_output, axis=0)
        input = np.matmul(self.w.val, pre_output) + self.bias.val
        p = dropout_probability
        dropout_mask = (np.random.random_sample(input.shape) < p) / p
        self.input.val = np.average(input * dropout_mask, axis=0)
        return self.forward_function(input)

    def backward(self, output_grad, learning_rate=1e-3):
        self.input.grad = self.backward_function(output_grad) * output_grad
        self.w.grad = np.dot(self.input.grad, self.pre_output.val.T)
        self.w.grad += self.w.alpha * self.w.val
        self.bias.grad = self.input.grad
        self.bias.grad += self.bias.alpha * self.bias.val
        self.pre_output.grad = np.dot(self.w.val.T, self.input.grad)

        for elem in self.update_list:
            elem.t += 1
            elem.first_moment = self.beta1 * elem.first_moment + (1 - self.beta1) * elem.grad
            elem.second_moment = self.beta2 * elem.second_moment + (1 - self.beta2) * np.sum(np.square(elem.grad))
            first_unbias = elem.first_moment / (1 - self.beta1 ** elem.t)
            second_unbias = elem.second_moment / (1 - self.beta2 ** elem.t)
            elem.val -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7)

        return np.copy(self.pre_output.grad)


class NeuralNetwork:
    def __init__(self,
                 learning_rate=5e-4, alpha=1, beta1=0.9, beta2=0.999,
                 loss_func=dis,
                 input_size=input_size,
                 hidden_size=(2, 2),
                 output_size=output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = len(hidden_size) + 1
        self.input = np.zeros((input_size, 1))
        self.output = np.zeros((output_size, 1))
        layer_list = (input_size, ) + hidden_size + (output_size, )
        self.layer_list = []
        for i in range(0, self.layers):
            self.layer_list.append(NeuralLayer(
                pre_size=layer_list[i],
                size=layer_list[i+1],
                activation_name='leaky_relu',
                alpha=alpha,
                beta1=beta1,
                beta2=beta2)
            )

        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.log_loss = []

    def forward(self, input, dropout_probability=1):
        self.input = np.average(input, axis=0)
        output = np.copy(input)
        for layer in self.layer_list:
            output = layer.forward(output, dropout_probability)
        self.output = np.average(output, axis=0)
        return output

    def backward(self, output_grad, learning_rate=1e-3):
        for layer in self.layer_list[::-1]:
            output_grad = layer.backward(output_grad, learning_rate)

    def train(self, samples, labels, batch=1024, epoch=50):
        num_samples = samples.shape[0]
        cnt = 1
        while True:
            per = np.asarray(range(num_samples))
            np.random.shuffle(per)
            start_index = 0
            while start_index < num_samples:
                end_index = min(start_index + batch, num_samples)
                select_index = per[start_index: end_index]
                select_samples = np.reshape(samples[select_index], (-1, self.input_size, 1))
                select_labels = np.reshape(labels[select_index], (-1, self.output_size, 1))
                samples_output = self.forward(select_samples, 0.5)
                grad_o = np.average(samples_output - select_labels, axis=0)
                loss_value = np.average(self.loss_func(samples_output, select_labels), axis=0)
                for layer in self.layer_list:
                    loss_value += layer.regular_loss()
                grad_o /= loss_value

                if math.isnan(loss_value) or math.isinf(loss_value):
                    exit(0)
                self.backward(grad_o, self.learning_rate)
                start_index = end_index
                self.log_loss.append(loss_value)
                print('[%d/%d] epoch [%d/%d] batches, loss value is %f' %
                      (cnt, epoch, start_index, num_samples, loss_value))
            cnt += 1
            if cnt > epoch:
                break

        self.log_loss = np.asarray(self.log_loss)
        log_cnt = np.asarray(range(1, self.log_loss.shape[0] + 1))

        plt.figure()
        plt.scatter(log_cnt, self.log_loss)
        plt.xlabel('cnt')
        plt.ylabel('loss')
        plt.title('dataset')
        plt.show()

    def pred(self, tests):
        return self.forward(tests)


if __name__ == '__main__':
    nn = NeuralNetwork(
        learning_rate=5e-4,
        alpha=0,
        beta1=0.9,
        beta2=0.999,
        input_size=input_size,
        output_size=output_size,
        hidden_size=(10, 10)
    )
    train_size = 100000
    x = np.random.random_sample((train_size, input_size, 1))
    y = np.sum(x, axis=1).reshape((train_size, output_size, 1))
    nn.train(x, y)

    # test_size = 1
    # test_x = np.array([[[0.3], [0.2]]])
    # test_y = nn.pred(test_x)

    # for i in range(test_size):
    #     print('[x, y]= ', test_x[i])
    #     print('[pred]= ', test_y[i])
    #     print()

    test_size = 10
    test_x = np.random.random_sample((test_size, input_size, 1))
    label_x = np.sum(test_x, axis=1).reshape((test_size, 1, 1))
    test_y = nn.pred(test_x)
    loss_lst = dis(label_x, test_y)

    for i in range(test_size):
        x = np.copy(test_x[i])
        y = np.copy(test_y[i])
        x = x.reshape((1, input_size))
        y = y.reshape((1, output_size))
        print(i, '[test, pred]=[', x, ',', y, ']', 'loss=', loss_lst[i])

    exit(0)
