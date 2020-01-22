# -*- coding:utf-8 -*-
__author__ = 'Wang'

import os
import heapq
import multiprocessing
import gensim
import logging
import json
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from pylab import *
from gensim.models import word2vec
from tflearn.data_utils import pad_sequences
from tqdm import tqdm

# TEXT_DIR = '../data/content.txt'
# METADATA_DIR = '../data/metadata.tsv'


def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def create_prediction_file(output_file, data_id, all_labels, all_predict_labels, all_predict_scores):
    """
    Create the prediction file.

    Args:
        output_file: The all classes predicted results provided by network
        data_id: The data record id info provided by class Data
        all_labels: The all origin labels
        all_predict_labels: The all predict labels by threshold
        all_predict_scores: The all predict scores by threshold
    Raises:
        IOError: If the prediction file is not a <.json> file
    """
    if not output_file.endswith('.json'):
        raise IOError("✘ The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        data_size = len(all_predict_labels)
        for i in range(data_size):
            predict_labels = [int(i) for i in all_predict_labels[i]]
            predict_scores = [round(i, 4) for i in all_predict_scores[i]]
            labels = [int(i) for i in all_labels[i]]
            data_record = OrderedDict([
                ('id', data_id[i]),
                ('labels', labels),
                ('predict_labels', predict_labels),
                ('predict_scores', predict_scores)
            ])
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def get_onehot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted onehot labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        onehot_labels_list = [0] * len(score)
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                onehot_labels_list[index] = 1
                count += 1
        if count == 0:
            max_score_index = score.index(max(score))
            onehot_labels_list[max_score_index] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_onehot_label_topk(scores, top_num=1):
    """
    Get the predicted onehot labels based on the topK number.

    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        onehot_labels_list = [0] * len(score)
        max_num_index_list = list(map(score.index, heapq.nlargest(top_num, score)))
        for i in max_num_index_list:
            onehot_labels_list[i] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_label_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Note: Only Used in `test_model.py`

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_labels: The predicted labels
        predicted_scores: The predicted scores
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        score_list = []
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                index_list.append(index)
                score_list.append(predict_score)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            score_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def get_label_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK number.
    Note: Only Used in `test_model.py`

    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        The predicted labels
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        score_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            score_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def load_word2vec_matrix(embedding_size,model,vocab):
    """
    Return the word2vec model matrix.

    Args:
        embedding_size: The embedding size
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    # word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'
    # word2vec_file = 'D:\work\多标签分类\Multi-Label-Text-Classification\data\word2vec_\word2vec_baike'
    # word2vec_file = '../data/char_vector/char_vector.npy'
    # vocab_path = '../data/char_vector/vob_2_int.npy'
    # if not os.path.isfile(word2vec_file):
    #     raise IOError("✘ The word2vec file doesn't exist. "
    #                   "Please use function <create_vocab_size(embedding_size)> to create it!")

    # model = np.load(word2vec_file)
    # vocab_file=np.load(vocab_path,allow_pickle=True).item()
    # vocab_size = len(vocab_file)
    vocab_size=len(vocab)
    # vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    # vocab = vocab_file
    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for key, value in vocab.items():
        if key is not None:
            embedding_matrix[value] = model[value]
    return vocab_size, embedding_matrix






def create_exmple(testid,features_content, labels_index):
    """
    构建的时候一般数据用 bytes_list，单个浮点数字使用 float_list，单个整数数字用int64_list。

    :param testid:
    :param features_content:
    :param labels_index
    :return:
    """
    # tf.train.Example -->  tf.train.Features --> tf.train.Feature
    return tf.train.Example(features=tf.train.Features(feature={
        'testid': tf.train.Feature(int64_list=tf.train.Int64List(value=[testid])),
        'features_content': tf.train.Feature(int64_list=tf.train.Int64List(value=features_content)),
        'labels_index': tf.train.Feature(int64_list=tf.train.Int64List(value=labels_index)),
    }))

def create_tfrecords(tfrecord_dir_path,path):
    """
    读取原数据集，写入到tfrecord文件中。
    :param tfrecord_dir_path:
    :param mnist_data_path:
    :return:
    """
    if not os.path.exists(tfrecord_dir_path):
        os.makedirs(tfrecord_dir_path)

    # 1\构建一个输出的文件对象。
    train_tfrecord_path = os.path.join(tfrecord_dir_path, 'train.tfrecord')
    test_tfrecord_path = os.path.join(tfrecord_dir_path, 'test.tfrecord')

    X,Y=get_data(path,size)
    x_train, y_train, x_test, y_test=split_data(X,Y,per)

    # 2、构建写入文件的writer对象
    with tf.python_io.TFRecordWriter(train_tfrecord_path) as train_writer:
        # 3、加载原数据
        # mnist = input_data.read_data_sets(train_dir=mnist_data_path,
        #                                   one_hot=False,
        #                                   validation_size=0)

        # 4、遍历所有数据进行输出操作。
        print('开始执行train data写入操作')
        for idx,lab in zip(x_train,y_train):
            image = idx
            label = lab

            #  a 需要将上述2个对象，转换为float32的数据。并且全部为1维的数组。
            image = np.reshape(image, -1).astype(np.float32)
            label = np.reshape(label, -1).astype(np.float32)

            #  b 将image 和label转换为Example对象。（转换之前需要用 tobytes()转换为二级制格式。）
            example = create_exmple(image.tobytes(), label.tobytes())

            # c 对example对象进行输出（输出序列化的值）
            train_writer.write(example.SerializeToString())  # 序列化为字符串

    with tf.python_io.TFRecordWriter(test_tfrecord_path) as test_writer:
        # 4、遍历所有数据进行输出操作。
        print('开始test data写入操作')
        for idx,lab in zip(x_test,y_test):
            image = idx
            label = lab

            #  a 需要将上述2个对象，转换为float32的数据。并且全部为1维的数组。
            image = np.reshape(image, -1).astype(np.float32)
            label = np.reshape(label, -1).astype(np.float32)

            #  b 将image 和label转换为Example对象。（转换之前需要用 tobytes()转换为二级制格式。）
            example = create_exmple(image.tobytes(), label.tobytes())

            # c 对example对象进行输出（输出序列化的值）
            test_writer.write(example.SerializeToString())  # 序列化为字符串



def data_augmented(data, drop_rate=1.0):
    """
    Data augmented.

    Args:
        data: The Class Data()
        drop_rate: The drop rate
    Returns:
        aug_data
    """
    aug_num = data.number
    aug_testid = data.testid
    aug_tokenindex = data.tokenindex
    aug_labels = data.labels
    aug_onehot_labels = data.onehot_labels
    aug_labels_num = data.labels_num

    for i in range(len(data.tokenindex)):
        data_record = data.tokenindex[i]
        if len(data_record) == 1:  # 句子长度为 1，则不进行增广
            continue
        elif len(data_record) == 2:  # 句子长度为 2，则交换两个词的顺序
            data_record[0], data_record[1] = data_record[1], data_record[0]
            aug_testid.append(data.testid[i])
            aug_tokenindex.append(data_record)
            aug_labels.append(data.labels[i])
            aug_onehot_labels.append(data.onehot_labels[i])
            aug_labels_num.append(data.labels_num[i])
            aug_num += 1
        else:
            data_record = np.array(data_record)
            for num in range(len(data_record) // 10):  # 打乱词的次数，次数即生成样本的个数；次数根据句子长度而定
                # random shuffle & random drop
                data_shuffled = np.random.permutation(np.arange(int(len(data_record) * drop_rate)))
                new_data_record = data_record[data_shuffled]

                aug_testid.append(data.testid[i])
                aug_tokenindex.append(list(new_data_record))
                aug_labels.append(data.labels[i])
                aug_onehot_labels.append(data.onehot_labels[i])
                aug_labels_num.append(data.labels_num[i])
                aug_num += 1

    class _AugData:
        def __init__(self):
            pass

        @property
        def number(self):
            return aug_num

        @property
        def testid(self):
            return aug_testid

        @property
        def tokenindex(self):
            return aug_tokenindex

        @property
        def labels(self):
            return aug_labels

        @property
        def onehot_labels(self):
            return aug_onehot_labels

        @property
        def labels_num(self):
            return aug_labels_num

    return _AugData()


def load_train_data_and_labels(data_file,pad_seq_len,tfrecord_dir_path):
    """
    Load research data from files, splits the data into words and generates labels.
    Return split sentences, labels and the max sentence length of the research data.

    Args:
        data_file: The research data
        num_labels: The number of classes
        embedding_size: The embedding size
        data_aug_flag: The flag of data augmented
    Returns:
        The class Data
    """
    # word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'
    word2vec_file = '../data/word_vec/word_vector.npy'
    vocab_file='../data/word_vec/vob_2_int.npy'
    # Load word2vec model file
    # if not os.path.isfile(word2vec_file):
    #     create_word2vec_model(embedding_size, TEXT_DIR)

    # model = word2vec.Word2Vec.load(word2vec_file)
    model = np.load(word2vec_file)
    # print(type(model[1][1]))
    vocab=np.load(vocab_file,allow_pickle=True).item()
    # print(model)
    # Load data from files and split by words
    #input_file, num_labels,word2vec_model,vocab_file,pad_seq_len,tfrecord_dir_path,is_training
    data = train_data_word2vec(
        input_file=data_file,
        # num_labels=num_labels,
        word2vec_model=model,
        vocab_file=vocab,
        pad_seq_len=pad_seq_len,
        tfrecord_dir_path=tfrecord_dir_path,
    )
    # if data_aug_flag:
    #     data = data_augmented(data)
    np.save('../data/new_word_vector',data.word2vec_model)
    # plot_seq_len(data_file, data)

    return data


def load_tfrecord(tfrecord_file_path, batch_size,pad_seq_len,num_labels):
    # 1、基于给定的文件获取一个队列（获取为string类型）
    producer = tf.train.string_input_producer([tfrecord_file_path])

    # 2、定义读取队列中数据的读取器。
    reader = tf.TFRecordReader()

    # 3、读取队列中的数据（定义一个操作）
    _, serialized_example = reader.read(queue=producer)

    # 4、对序列化之后的数据进行解析。
    features = tf.parse_single_example(serialized_example, features={
        "testid": tf.FixedLenFeature([], tf.int64),
      "features_content": tf.FixedLenFeature([pad_seq_len], tf.int64),
      "labels_index": tf.FixedLenFeature([num_labels], tf.int64),
    })

    # 5、重新转换格式。
    for name in list(features.keys()):
      t = features[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      features[name] = t
    testid=features['testid']
    features_content=features['features_content']
    labels_index=features['labels_index']
    batch_testid,batch_features_content,batch_labels_index= tf.train.shuffle_batch([testid,features_content,labels_index],batch_size=batch_size,
                                                                        num_threads=2,
                                                                        capacity=batch_size * 5,
                                                                        min_after_dequeue=batch_size *2)
    return batch_testid,batch_features_content,batch_labels_index

def _create_onehot_labels(labels_index,num_classes):
    label = [0] * num_classes
    print(labels_index)
    for item in labels_index:
        label[int(item)] = 1
    return label


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data: The data
        batch_size: The size of the data batch
        num_epochs: The number of epochs
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
