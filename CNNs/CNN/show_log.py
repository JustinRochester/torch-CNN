from .GPU_np import np
import matplotlib.pyplot as plt


def show_log(train_log, test_log, pred_log, acc_log):
    train_len = len(train_log)
    test_len = len(test_log)

    plt.subplot(1, 3, 1)
    plt.title('loss')
    x = np.arange(train_len) * (test_len - 1) / (train_len - 1)
    x = x.tolist()
    plt.plot(x, train_log, label='train_loss')
    x = np.arange(test_len).tolist()
    plt.plot(x, test_log, label='test_loss')
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.title('precision')
    x = np.arange(test_len).tolist()
    plt.plot(x, pred_log, label='test_precision')
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.title('accuracy')
    x = np.arange(test_len).tolist()
    plt.plot(x, acc_log, label='test_accuracy')
    plt.grid()

    plt.show()
