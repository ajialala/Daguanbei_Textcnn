# coding: utf-8

from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr

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
