#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time

import pandas as pd
import tensorflow as tf

from cnn_model import TCNNConfig, TextCNN
from helper import read_vocab, read_category, batch_iter, process_file, build_vocab, feed_data, evaluate
from helper import get_time_dif, load_model, load_data

base_dir = 'data'
train_dir = os.path.join(base_dir, 'train_data_w')
val_dir = os.path.join(base_dir, 'val_data_w')
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = 'checkpoints/textcnn/'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

temp_dir = 'temp/'               # 存放训练集和验证集的处理后文件


def train():
    # 配置 Tensorboard，每次训练的结果保存在以日期时间命名的文件夹中。
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = 'tensorboard/textcnn' + '/' + time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 载入训练集与验证集
    print("Loading training and validation data...")
    start_time = time.time()
    x_train, y_train, x_val, y_val = load_data(temp_dir, train_dir, val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    total_batch = tf.Variable(0, trainable=False) # 总批次，不可训练的变量

    # 创建session
    session = tf.Session()
    # 导入权重
    saver = load_model(session, save_dir)
    # 图写入tensorboard
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = session.run(total_batch)  # 记录上一次提升批次
    require_improvement = 5000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)

        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(model, x_batch, y_batch, config.dropout_keep_prob)

            if session.run(total_batch) % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, session.run(total_batch))

            if session.run(total_batch) % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                loss_train, F1_train, _, _ = evaluate(session, model, x_train, y_train)
                loss_val, F1_val, _, _ = evaluate(session, model, x_val, y_val)  # todo

                if F1_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = F1_val
                    last_improved = session.run(total_batch)
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train F1: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val F1: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(session.run(total_batch), loss_train, F1_train, loss_val, F1_val, time_dif, improved_str))


            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            session.run(tf.assign(total_batch, total_batch+1))   # 用tf.assign迭代total_batch可以在saver中记录total_batch的变化

            if session.run(total_batch) - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break
    session.close()


if __name__ == '__main__':

    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories = [str(a) for a in list(set(pd.read_csv(train_dir,header=None,sep='\t')[0]))]
    cat_to_id = read_category(categories)
    words, word_to_id = read_vocab(vocab_dir)
    if config.vocab_size != len(words):
        print('Your vocab_size of config is different from vocab_size of vocab_file！')
        print('You can clean vocab_file or reset vocab_size of config.')
        sys.exit()
    model = TextCNN(config)

    train()
