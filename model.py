from numpy.random import seed

seed(1)
import tensorflow as tf

tf.random.set_seed(2)
import os
import numpy as np
import argparse
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# saved_model_path = "./save_model/"


def read_dataset(train_path, test_path):
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)
    print(train_data.shape)

    train_ids = train_data[:, 79:]

    test_ids = test_data[:, 79:]
    # print(train_ids.shape)

    train_data = train_data[:, :79]
    test_data = test_data[:, :79]
    # print(train_data.shape)
    return train_data.astype(np.float32), train_ids, test_data.astype(np.float32), test_ids


def read_data_t(test_path):
    test_data = np.load(test_path, allow_pickle=True)
    print(test_data.shape)
    test_ids = test_data[:, 79:]
    test_data = test_data[:, :79]
    return test_data.astype(np.float32), test_ids


def plotCM(classes, matrix):
    """classes: a list of class names"""
    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    # plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)

    plt.show()


def xgboost():
    model = xgb.XGBClassifier(random_state=66, use_label_encoder=False, n_estimators=100, max_depth=13,
                              learning_rate=0.01)
    train_data, train_ids, test_data, test_ids = read_dataset(train_path, test_path)
    model.fit(train_data, train_ids)
    y_pred = model.predict(test_data)
    result = confusion_matrix(test_ids, y_pred)
    print(result)
    acc = accuracy_score(test_ids, y_pred)
    print(acc)
    result1 = classification_report(test_ids, y_pred)
    print(result1)

    plotCM(['benign', 'SCF', 'VPS', 'DNS', 'DoH', 'SCF_CC', 'VPS_CC', 'DNS_CC', 'DoH_CC'], result)
    # model.save_model('model.bin')


def svm_train():
    model = SGDClassifier(random_state=66)
    train_data, train_ids, test_data, test_ids = read_dataset(train_path, test_path)
    model.fit(train_data, train_ids)
    y_pred = model.predict(test_data)
    result = confusion_matrix(test_ids, y_pred)
    print(result)
    acc = accuracy_score(test_ids, y_pred)
    print(acc)
    result1 = classification_report(test_ids, y_pred)
    print(result1)


def RF_train():
    classifier = RandomForestClassifier(n_estimators=75, criterion='gini', random_state=78)
    train_data, train_ids, test_data, test_ids = read_dataset(train_path, test_path)
    classifier.fit(train_data, train_ids)
    y_pred = classifier.predict(test_data)
    result = confusion_matrix(test_ids, y_pred)
    print(result)
    acc = accuracy_score(test_ids, y_pred)
    print(acc)
    result1 = classification_report(test_ids, y_pred)
    print(result1)

def stacking_train():
    clf1 = RandomForestClassifier(n_estimators=75, criterion='gini', random_state=78)
    clf2 = SGDClassifier(random_state=66)
    clf3 = xgb.XGBClassifier(random_state=66, use_label_encoder=False, n_estimators=100, max_depth=12,
                              learning_rate=0.01)
    train_data, train_ids, test_data, test_ids = read_dataset(train_path, test_path)




def xgb_test():
    model = xgb.Booster()

    print("load model.....")
    model.load_model('model.bin')
    test_data, test_ids = read_data_t(test_path)
    dtest = xgb.DMatrix(test_data)
    y_pred = model.predict(dtest, output_margin=True)

    y_pred = y_pred.argmax(axis=1)

    # print(y_pred[0])
    result = confusion_matrix(test_ids, y_pred)
    print(result)
    acc = accuracy_score(test_ids, y_pred)
    print(acc)
    result1 = classification_report(test_ids, y_pred)
    print(result1)

    plotCM(['benign', 'SCF', 'VPS', 'DNS', 'DoH', 'SCF_CC', 'VPS_CC', 'DNS_CC', 'DoH_CC'], result)


if __name__ == '__main__':
    print("hello world")
    # 训练模型并测试
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('test_arg',
    #                     help='测试集数据(.npy)')
    # parser.add_argument('--mode2', action='store_true',
    #                     help='进行训练再测试(会覆盖原来保存的模型)')
    # parser.add_argument('train_arg', nargs='?', default=None,
    #                     help='训练集数据(.npy)')
    # 读取已完成训练的模型单独进行测试
    # args = parser.parse_args()
    # test_path = args.test_arg
    # train_path = args.train_arg
    # if args.mode2:
    #     print("重新进行训练，并进行测试，会覆盖旧的模型数据")
    #     print("--------------------------------------")
    #     xgboost()
    # else:
    #     print("--------------------------------------")
    #     print("直接进行测试")
    #     xgb_test()
    # xgb_test()
    train_path = "./810train_data.npy"
    test_path = "./810test_data.npy"
    xgboost()

