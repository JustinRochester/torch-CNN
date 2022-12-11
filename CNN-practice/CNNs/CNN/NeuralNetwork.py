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
                 optimizer_name='Adam',
                 alpha=0.1,
                 learning_rate_function=(lambda lr, i, tot: lr)
                 ):
        super().__init__()
        self.layers = []
        self.input_size = input_size
        self.output_size = (class_num, 1)
        self.learning_rate = learning_rate
        self.loss_function = loss_dict[loss_name]
        self.optimizer = None
        self.alpha = alpha
        self.learning_rate_function = learning_rate_function

        self.train_log = []
        self.test_log = []
        self.pred_log = []
        self.acc_log = []
        self.optimizer = optimizer_dict[optimizer_name]()
        self.recorder = Recorder()

    def get_data(self):
        return super().get_data() + self.optimizer.get_data()

    def set_data(self, data_iter):
        super().set_data(data_iter)
        self.optimizer.set_data(data_iter)

    def save(self, version, save_path):
        self.recorder.set_path(save_path)
        data = self.get_data()
        self.recorder.save_version(version, data)
        self.recorder.save_log(version, self.train_log, self.test_log, self.pred_log, self.acc_log)

        if max(self.acc_log) != self.acc_log[-1] and self.recorder.exists_best():
            return
        self.recorder.save_best(data)

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
            data = self.recorder.load_version(version)
            self.set_data(iter(data))
            self.train_log, self.test_log, self.pred_log, self.acc_log = self.recorder.load_log(version)
        else:
            data = self.recorder.load_best()
            self.set_data(iter(data))

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
            regular_loss=self.optimizer.regular_loss()
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
                    epoch_id=1, epoch_number=1):
        n = image_array.shape[0]
        select_index = np.random.permutation(n)
        for l_batch in range(0, n, batch_size):
            r_batch = min(n, l_batch + batch_size)
            self.zero_grad()
            loss_sum = 0
            output = self.forward(image_array[select_index[l_batch: r_batch]])
            loss_value, output_grad = self.loss_function.loss(
                label=label_array[select_index[l_batch: r_batch]],
                output=output,
                regular_loss=self.optimizer.regular_loss()
            )
            self.backward(output_grad)
            loss_sum += np.sum(loss_value)
            self.multi_grad(multiply=1/(r_batch-l_batch+1))
            self.optimizer.update(self.learning_rate)
            self.train_log_record(epoch_id, epoch_number, r_batch, n, float(loss_sum) / (r_batch - l_batch + 1))

    def train(self, image_array, label_array, epoch_number=50,
              batch_size=1024,
              test_image_array=None, test_label_array=None,
              version=0,
              save_path=None):
        self.load(version=version, save_path=save_path)
        if version == 0:
            self.test_log_record(version, epoch_number, test_image_array, test_label_array, batch_size)
        for i in range(version + 1, epoch_number + 1):
            start_time = time.perf_counter()
            self.train_epoch(image_array, label_array, batch_size, i, epoch_number)
            end_time = time.perf_counter()
            self.learning_rate = self.learning_rate_function(self.learning_rate, i, epoch_number)

            message = "finished training in [%d/%d] epoch, cost time = %f second(s)"
            message = message % (i, epoch_number, end_time - start_time)
            print(message)
            self.save(version=i, save_path=save_path)
            self.test_log_record(i, epoch_number, test_image_array, test_label_array, batch_size)
        show_log(self.train_log, self.test_log, self.pred_log, self.acc_log)
