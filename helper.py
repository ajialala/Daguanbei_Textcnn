# coding: utf-8

import os
import time
import pickle
from collections import Counter
from datetime import timedelta

import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr
from sklearn import metrics

def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                content = content.split(' ')
                if content:
                    #contents.append(list(native_content(content)))
                    contents.append(content)
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category(categories):
    """读取分类目录，固定"""

    cat_to_id = dict(zip(categories, range(len(categories))))

    return cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        # yield 的作用就是把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数，
        # Python 解释器会将其视为一个 generator，调用 batch_iter 不会执行 batch_iter 函数，
        # 而是返回一个 iterable 对象！在 for 循环执行时，每次循环都会执行 batch_iter 函数内部的代码，
        # 执行到 yield 时，batch_iter 函数就返回一个迭代值，下次迭代时，代码从 yield 的下一条语句继续执行，
        # 而函数的本地变量看起来和上次中断执行前是完全一样的，于是函数继续执行，直到再次遇到 yield。
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def feed_data(model, x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, model,  x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    i = 0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(model, x_batch, y_batch, 1.0)
        if i == 0:
            loss, y_pre = sess.run([model.loss, model.y_pred_cls], feed_dict=feed_dict)
            y_true = y_batch
        else:

            loss, y_tem = sess.run([model.loss, model.y_pred_cls], feed_dict=feed_dict)
            y_pre = np.hstack((y_pre, y_tem))
            y_true = np.vstack((y_true, y_batch))
        total_loss += loss * batch_len
        i += 1

    F1 = metrics.f1_score(sess.run(tf.argmax(y_true, 1)), y_pre, average='macro')
    precision = metrics.precision_score(sess.run(tf.argmax(y_true, 1)), y_pre, average='macro')
    recall = metrics.recall_score(sess.run(tf.argmax(y_true, 1)), y_pre, average='macro')

    return total_loss / data_len, F1, precision, recall


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def load_model(sess, path):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        print("Could not find old weights!")
    return saver


def load_data(temp_dir, train_dir, val_dir, word_to_id, cat_to_id, seq_length):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not os.path.exists(temp_dir+'x_train.pkl'):
        x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
        x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
        pickle.dump(x_train, open(temp_dir+'x_train.pkl', 'wb'))
        pickle.dump(y_train, open(temp_dir+'y_train.pkl', 'wb'))
        pickle.dump(x_val, open(temp_dir+'x_val.pkl', 'wb'))
        pickle.dump(y_val, open(temp_dir+'y_val.pkl', 'wb'))
    else:
        x_train = pickle.load(open(temp_dir+'x_train.pkl', 'rb'))
        y_train = pickle.load(open(temp_dir+'y_train.pkl', 'rb'))
        x_val = pickle.load(open(temp_dir+'x_val.pkl', 'rb'))
        y_val = pickle.load(open(temp_dir+'y_val.pkl', 'rb'))

    return x_train, y_train, x_val, y_val