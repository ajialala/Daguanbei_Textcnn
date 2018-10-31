# -*- coding: utf-8 -*-
import os
import time
import pickle
import tensorflow as tf

from cnn_model import TCNNConfig, TextCNN
from helper import get_time_dif, load_model, evaluate
from run_cnn import eval_save_dir, temp_dir

# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率。
EVAL_INTERVAL_SECS = 10


def main():
    # 载入训练集与验证集
    print("Loading training and validation data...")
    start_time = time.time()
    x_train = pickle.load(open(temp_dir + 'x_train.pkl', 'rb'))
    y_train = pickle.load(open(temp_dir + 'y_train.pkl', 'rb'))
    x_val = pickle.load(open(temp_dir + 'x_val.pkl', 'rb'))
    y_val = pickle.load(open(temp_dir + 'y_val.pkl', 'rb'))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    # 导入权重
    saver = load_model(session, eval_save_dir)

    print('Evaluating...')
    start_time = time.time()
    best_acc_val = 0.0  # 最佳验证集准确率

    # 每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化。
    while True:
        # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名。
        ckpt = tf.train.get_checkpoint_state(eval_save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # 加载模型
            saver.restore(session, ckpt.model_checkpoint_path)
            # 通过文件名得到模型保存时迭代的轮数
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            loss_train, F1_train, _, _ = evaluate(session, model, x_train, y_train)
            loss_val, F1, _, _ = evaluate(session, model, x_val, y_val)
            if F1 > best_acc_val:
                best_acc_val = F1
                improved_str = '*'
            else:
                improved_str = ''

            time_dif = get_time_dif(start_time)
            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train F1: {2:>7.2%},' \
                  + ' Val Loss: {3:>6.2}, Val F1: {4:>7.2%}, Time: {5} {6}'
            print(msg.format(global_step, loss_train, F1_train, loss_val, F1, time_dif, improved_str))
        else:
            print('No checkpoint file found')
            return
        time.sleep(EVAL_INTERVAL_SECS)

    session.close()


if __name__ == '__main__':
    config = TCNNConfig()
    model = TextCNN(config)
    main()
