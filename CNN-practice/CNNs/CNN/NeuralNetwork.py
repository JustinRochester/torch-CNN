import time
from .Base.softmax import softmax
from .GPU_np import np
from .show_log import show_log
from .Loss import loss_dict
from .Optimizers import optimizer_dict
from .Base.labels2onehot import labels2onehot, probability2labels
from .Sequential import Sequential
from .Recorder import Recorder
from .Layers import *


class NeuralNetwork(Sequential):
    def __init__(self,
                 input_size=(3, 32, 32),
                 class_num=2,
                 learning_rate=1e-2,
                 loss_name='MSE',
                 alpha=0.1,
                 learning_rate_function=(lambda lr, i, tot: lr)
                 ):
        super().__init__()
        self.layers = []
        self.input_size = input_size
        self.output_size = (class_num, 1)
        self.learning_rate = learning_rate
        self.loss_function = loss_dict[loss_name]
        self.alpha = alpha
        self.learning_rate_function = learning_rate_function

        self.train_log = []
        self.test_log = []
        self.pred_log = []
        self.acc_log = []
        self.optimizer_dict = {name: optimizer_dict[name]() for name in optimizer_dict.keys()}
        self.recorder = Recorder()

    def save(self, version, save_path):
        self.recorder.set_path(save_path)
        self.recorder.save_version(version, self.optimizer_dict)
        self.recorder.save_log(version, self.train_log, self.test_log, self.pred_log, self.acc_log)

        if max(self.acc_log) != self.acc_log[-1] and self.recorder.exists_best():
            return
        self.recorder.save_best(self.optimizer_dict)

    def load(self, version, save_path):
        self.recorder.set_path(save_path)
        self.train_log = []
        self.test_log = []
        self.pred_log = []
        self.acc_log = []
        if save_path is None:
            return
        if version == 0:
            self.recorder.remove_best()
            return
        if version != 'best':
            self.optimizer_dict = self.recorder.load_version(version)
            self.train_log, self.test_log, self.pred_log, self.acc_log = self.recorder.load_log(version)
        else:
            self.optimizer_dict = self.recorder.load_best()

        for optimizer_name in self.optimizer_dict.keys():
            self.optimizer_dict[optimizer_name].load_data()
        optimizer_iter_list = {
                            optimizer_name: self.optimizer_dict[optimizer_name].get_iter()
                            for optimizer_name in optimizer_dict.keys()
                          }
        for layer in self.layers:
            layer.load_model(optimizer_iter_list)

    def regular_loss(self):
        return sum([optimizer.regular_loss() for optimizer in self.optimizer_dict.values()])

    def zero_grad(self):
        for optimizer in self.optimizer_dict.values():
            optimizer.zero_grad()

    def multi_grad(self, multiply=1):
        for optimizer in self.optimizer_dict.values():
            optimizer.multi_grad(multiply)

    def optimizer_update(self):
        for optimizer in self.optimizer_dict.values():
            optimizer.update(self.learning_rate)

    def predict(self, test_images, batch_size=128, test_pred_need=False, test_loss_need=False, test_labels=None):
        n = test_images.shape[0]
        output_array = np.empty((n,) + self.output_size)
        for l in range(0, n, batch_size):
            r = min(n, l + batch_size)
            output_array[l: r] = self.predict_forward(test_images[l: r])
        test_pred = softmax(output_array)

        predict_labels = probability2labels(test_pred)
        predict_onehot = labels2onehot(predict_labels, self.output_size[0]).reshape(test_pred.shape)
        precision_array = test_pred.reshape((n, -1))[np.arange(n), predict_labels].reshape(n)
        if test_loss_need is False or test_labels is None:
            if test_pred_need is False:
                return predict_onehot, precision_array
            else:
                return test_pred, predict_onehot, precision_array

        loss_array, _ = self.loss_function.loss(
            label=test_labels,
            output=output_array,
            regular_loss=self.regular_loss()
        )
        loss_array = loss_array.reshape(n)
        if test_pred_need is False:
            return predict_onehot, precision_array, loss_array
        else:
            return test_pred, predict_onehot, precision_array, loss_array

    def test_accuracy(self, test_images, test_labels, batch_size=128):
        n = test_images.shape[0]
        predict_onehot, _ = self.predict(test_images, batch_size)
        return float(1 - np.sum(np.square(test_labels - predict_onehot) / (2 * n)))

    def evaluate(self, test_images, test_labels, batch_size=128):
        n = test_images.shape[0]
        test_pred, predict_onehot, precision_array, loss_array = self.predict(
            test_images=test_images,
            batch_size=batch_size,
            test_pred_need=True,
            test_loss_need=True,
            test_labels=test_labels
        )
        precision = float(np.sum(test_pred * test_labels) / n)
        loss_avg = float(np.average(loss_array))
        accuracy = float(1 - np.sum(np.square(test_labels - predict_onehot)) / (2 * n))
        return precision, loss_avg, accuracy

    def train_log_record(self, epoch_id, epoch_number, batch_id, batch_number, loss_value):
        message = "training in [%d/%d] epoch [%d/%d] batch, loss value = %f"
        message = message % (epoch_id, epoch_number, batch_id, batch_number, loss_value)
        print(message)
        self.train_log.append(loss_value)

    def test_log_record(self, epoch_id, epoch_number, test_image_array=None, test_label_array=None, batch_size=128):
        if test_image_array is None or test_label_array is None:
            return
        precision, loss_value, accuracy = self.evaluate(test_image_array, test_label_array, batch_size)
        self.pred_log.append(precision * 100)
        self.acc_log.append(accuracy * 100)
        self.test_log.append(loss_value)
        message = "training in [%d/%d] epoch, test loss = %f, precision = %f%%, accuracy = %f%%"
        message = message % (epoch_id, epoch_number, loss_value, precision * 100, accuracy * 100)
        print(message)

    def train_epoch(self, image_array, label_array,
                    batch_size=128,
                    run_size=32,
                    epoch_id=1, epoch_number=1):
        n = image_array.shape[0]
        select_index = np.random.permutation(n)
        for l_batch in range(0, n, batch_size):
            r_batch = min(n, l_batch + batch_size)
            self.zero_grad()
            loss_sum = 0
            for l_run in range(l_batch, r_batch, run_size):
                r_run = min(r_batch, l_run + run_size)
                output = self.forward(image_array[select_index[l_run: r_run]])
                loss_value, output_grad = self.loss_function.loss(
                    label=label_array[select_index[l_run: r_run]],
                    output=output,
                    regular_loss=self.regular_loss()
                )
                self.backward(output_grad)
                loss_sum += np.sum(loss_value)
            self.multi_grad(multiply=1/(r_batch-l_batch+1))
            self.optimizer_update()
            self.train_log_record(epoch_id, epoch_number, r_batch, n, float(loss_sum) / (r_batch - l_batch + 1))

    def train(self, image_array, label_array, epoch_number=50,
              batch_size=1024,
              run_size=32,
              test_image_array=None, test_label_array=None,
              version=0,
              save_path=None):
        self.load(version=version, save_path=save_path)
        if version == 0:
            self.test_log_record(version, epoch_number, test_image_array, test_label_array, run_size)
        for i in range(version + 1, epoch_number + 1):
            start_time = time.clock()
            self.train_epoch(image_array, label_array, batch_size, run_size, i, epoch_number)
            end_time = time.clock()
            self.learning_rate = self.learning_rate_function(self.learning_rate, i, epoch_number)

            message = "finished training in [%d/%d] epoch, cost time = %f second(s)"
            message = message % (i, epoch_number, end_time - start_time)
            print(message)
            self.save(version=i, save_path=save_path)
            self.test_log_record(i, epoch_number, test_image_array, test_label_array, batch_size)
        show_log(self.train_log, self.test_log, self.pred_log, self.acc_log)
