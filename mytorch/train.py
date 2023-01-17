import os
from time import perf_counter as clock
import nptorch
from nptorch.GPU_np import np
import numpy as cpu_np
from nptorch import nn
from nptorch.util import DataLoader, Recorder, Logger
import matplotlib as plt


def train_epoch(
        turn: int, total_turn: int,
        net: nn.Module,
        loss_function: nn.LossFunction,
        optimizer: nn.Optimizer,
        train_data: DataLoader,
        logger: Logger):
    epoch_start_time = clock()
    net.train_mode()
    finished = 0
    length = train_data.len
    for images, labels in train_data:
        now = train_data.select_position
        predict = net(images)
        loss = loss_function(predict, labels)
        loss.backward()
        optimizer.step()
        loss.zero_grad()
        average_loss = float(loss.data) / (now - finished)
        logger.append_loss(average_loss)
        print('epoch {}/{} batch {}/{}, loss={}'.format(turn, total_turn, now, length, average_loss))
        finished = now
    print('epoch {}/{} cost time: {} second(s)'.format(turn, total_turn, clock() - epoch_start_time))


def evaluate(net: nn.Module, data: DataLoader):
    if data is None:
        return None
    net.predict_mode()
    acc = 0
    for images, labels in data:
        predict = net(images).data
        predict = predict.argmax(axis=1)
        acc += np.sum(predict == labels.data)
    return int(acc) * 100 / data.len


def get_data(read_data, class_num=10, train_batch=32, test_batch=64, train_data_to_test=False):
    train_images, train_labels, test_images, test_labels = read_data()
    if train_data_to_test:
        train_evaluate_data = DataLoader(train_images, train_labels, batch_size=test_batch, shuffle=False)
        test_evaluate_data = DataLoader(test_images, test_labels, batch_size=test_batch, shuffle=False)
        evaluate_data = (train_evaluate_data, test_evaluate_data)
    else:
        test_evaluate_data = DataLoader(test_images, test_labels, batch_size=test_batch, shuffle=False)
        evaluate_data = (None, test_evaluate_data)

    tmp = cpu_np.zeros((train_images.shape[0], class_num))
    tmp[cpu_np.arange(train_images.shape[0]), train_labels] = 1
    train_labels = tmp
    train_data = DataLoader(train_images, train_labels, batch_size=train_batch)
    return *evaluate_data, train_data


def load_version(net: nn.Module, optimizer: nn.Optimizer, recorder: Recorder, version=0):
    if version == 0:
        return [], [], []
    if type(version) is not int:
        if type(version) is str and version.lower() == 'best':
            data_iter = iter(recorder.load_best())
        else:
            raise ValueError("Unexpected version")
    else:
        data_iter = iter(recorder.load_version(version))
    net.load_data_list(data_iter)
    optimizer.load_data_list(data_iter)
    if type(version) is int:
        return recorder.load_log(version)
    else:
        return [], [], []


def train(
        net: nn.Module,
        loss_function: nn.LossFunction,
        optimizer: nn.Optimizer,
        train_data: DataLoader,
        recorder: Recorder,
        test_data: tuple,
        train_data_to_test: bool = False,
        version: int = 0,
        epoch_number: int = 10,
    ):
    logs = load_version(net, optimizer, recorder, version)
    logger = Logger(train_data_to_test)
    logger.load_data(logs)
    train_evaluate_data, test_evaluate_data = test_data
    # if version == 0:
    #     logger.append_train_acc(evaluate(net, train_evaluate_data))
    #     logger.append_test_acc(evaluate(net, test_evaluate_data))

    logger.start_show()
    for i in range(version + 1, epoch_number + 1):
        train_epoch(
            turn=i,
            total_turn=epoch_number,
            net=net,
            loss_function=loss_function,
            optimizer=optimizer,
            train_data=train_data,
            logger=logger
        )
        recorder.save_version(
            version=i,
            data=net.get_data_list() + optimizer.get_data_list()
        )

        train_acc = 0
        if train_data_to_test:
            now = clock()
            train_acc = evaluate(net, train_evaluate_data)
            logger.append_train_acc(train_acc)
            print('cost time in predict train_data: {} second(s)'.format(clock() - now))

        now = clock()
        test_acc = evaluate(net, test_evaluate_data)
        if i == 1 or test_acc > max(logger.test_acc_log):
            recorder.save_best(net.get_data_list() + optimizer.get_data_list())
        logger.append_test_acc(test_acc)
        print('cost time in predict test_data: {} second(s)'.format(clock() - now))

        if train_data_to_test:
            msg = 'epoch {}/{}, train data accuracy={}, test data accuracy={}'
            msg = msg.format(i, epoch_number, train_acc, test_acc)
        else:
            msg = 'epoch {}/{}, test data accuracy={}'
            msg = msg.format(i, epoch_number, test_acc)
        recorder.save_log(i, *logger.get_data())
        print(msg)
    logger.stop_show()
