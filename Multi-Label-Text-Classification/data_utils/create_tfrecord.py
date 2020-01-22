# -*- coding:utf-8 -*-


import numpy as np
import tensorflow as tf
from tqdm import tqdm
import json
import os


def pad_sequence(content_id, maxlen, value=0):
    """
    样本padding,不足maxlen的补0，超过的截断
    :param content_id:
    :param maxlen:
    :param value:
    :return:
    """
    while len(content_id) < maxlen:
        content_id.append(value)
    return content_id[:maxlen]

def create_onehot_labels(labels_index,num_labels):
    """
    one_hot编码
    :param labels_index:
    :param num_labels:
    :return:
    """
    label = [0] * num_labels
    for item in labels_index:
        label[int(item)] = 1
    return label

def create_exmple(testid,features_content, labels_index):
    """
    tf.train.Example -->  tf.train.Features --> tf.train.Feature
    :param testid:
    :param features_content:
    :param labels_index:
    :return:
    """
    return tf.train.Example(features=tf.train.Features(feature={
        'testid': tf.train.Feature(int64_list=tf.train.Int64List(value=[testid])),
        'features_content': tf.train.Feature(int64_list=tf.train.Int64List(value=features_content)),
        'labels_index': tf.train.Feature(int64_list=tf.train.Int64List(value=labels_index)),
    }))

def train_data_word2vec(input_file,word2vec_model,vocab_file,pad_seq_len,tfrecord_dir_path,num_labels,vocab_embedding_dim=300):
    """
    将训练样本写入tfrecord中，对于不在词典中的单词随机初始化
    :param input_file:          训练样本路径
    :param word2vec_model:      embedding模型
    :param vocab_file:          词典
    :param pad_seq_len:         每个样本最大长度
    :param tfrecord_dir_path:   tfrecord路径
    :param num_labels:          label个数
    :param vocab_embedding_dim: 词向量维度
    :return:
    """
    vocab={} if vocab_file==None else vocab_file
    embedding_dim=vocab_embedding_dim if word2vec_model.any()==None else word2vec_model.shape[1]
    # print('The length of vocab：{}'.format(len(vocab)))
    def train_token_to_index(content,extra_word_vec,embedding_dim):
        result = []
        for item in content:
            word2id = vocab.get(item)
            if word2id is None:
                vocab[item]=len(vocab)
                word2id = len(vocab)-1
                new_vec=np.random.normal(0,0.1,[embedding_dim]).astype(np.float32)
                extra_word_vec.append(new_vec)
            result.append(word2id)
        return result,extra_word_vec
    if not input_file.endswith('.json'):
        raise IOError("✘ The research data is not a json file. "
                      "Please preprocess the research data into the json file.")
    tfrecord_path = os.path.join(tfrecord_dir_path, 'train.tfrecord')
    with tf.python_io.TFRecordWriter(tfrecord_path) as train_writer:
        print('Now start writing training data')
        with open(input_file,'r',encoding='utf-8') as f:
            total_line = 0
            extra_word_vec = []
            # embedding_dim = word2vec_model.shape[1]
            f=f.readlines()
            total_num=len(f)
            for eachline in tqdm(f,total=total_num):
                data = json.loads(eachline)
                testid = data['testid']
                features_content = data['features_content']
                labels_index = data['labels_index']
                content_id, extra_word_vec = train_token_to_index(features_content, extra_word_vec=extra_word_vec,embedding_dim=embedding_dim)
                pad_content_id = pad_sequence(content_id, maxlen=pad_seq_len, value=0)
                if len(pad_content_id)!=30:
                    print(pad_content_id)
                    print('haha')
                    break
                one_hot_label = create_onehot_labels(labels_index,num_labels)
                example = create_exmple(testid, pad_content_id,one_hot_label)
                train_writer.write(example.SerializeToString())
                total_line+=1

    model=np.array(extra_word_vec) if word2vec_model.any()==None else np.vstack([word2vec_model,np.array(extra_word_vec)])

    class _Data:
        def __init__(self):
            pass

        @property
        def number(self):
            return total_line

        @property
        def word2vec_model(self):
            return model

        @property
        def vocab(self):
            return vocab
    return _Data()


def load_train_data_and_labels(data_file,pad_seq_len,data_dir,num_labels):
    """
    将训练样本载入tfrecord中
    :param data_file:    训练样本路径
    :param pad_seq_len:  最大样本长度
    :param data_dir:     tfrecord写出路径
    :param num_labels:   类别个数
    :return:
    """
    print('loading word_vector and vocab')
    word2vec_file = os.path.join(data_dir,'word_vec/word_vector.npy')
    vocab_file=os.path.join(data_dir,'word_vec/vob_2_int.npy')
    # print(os.path.exists(word2vec_file))
    # print(os.path.exists(vocab_file))
    if not os.path.exists(word2vec_file) and not os.path.exists(vocab_file):
        print("未检测到预训练词向量，将初始化词向量")
        model=None
        vocab=None
    else:
        model = np.load(word2vec_file)
        vocab=np.load(vocab_file,allow_pickle=True).item()
    tfrecord_dir_path=os.path.join(data_dir,'tfrecord')
    if not os.path.exists(tfrecord_dir_path):
        os.mkdir(tfrecord_dir_path)
    data = train_data_word2vec(
        input_file=data_file,
        word2vec_model=model,
        vocab_file=vocab,
        pad_seq_len=pad_seq_len,
        tfrecord_dir_path=tfrecord_dir_path,
        num_labels=num_labels
    )
    new_word_vector=os.path.join(data_dir, 'word_vec/new_word_vector')
    new_vob_2_int=os.path.join(data_dir, 'word_vec/new_vob_2_int')
    np.save(new_word_vector,data.word2vec_model)
    np.save(new_vob_2_int,data.vocab)
    print('The number of training data:{}'.format(data.number))




def load_valid_data_and_labels(input_file,pad_seq_len,data_dir,num_labels):
    """
    将验证样本载入tfrecord中
    :param input_file:   验证样本路径
    :param pad_seq_len:  最大样本长度
    :param data_dir:     tfrecord写出路径
    :param num_labels:   类别个数
    :return:
    """
    vocab_file=os.path.join(data_dir,'word_vec/new_vob_2_int.npy')
    vocab=np.load(vocab_file,allow_pickle=True).item()
    tfrecord_dir_path=os.path.join(data_dir,'tfrecord')
    def _token_to_index(content):
        result = []
        for item in content:
            word2id = vocab.get(item)
            if word2id is None:
                word2id = 0
            result.append(word2id)
        return result

    if not input_file.endswith('.json'):
        raise IOError("✘ The research data is not a json file. "
                      "Please preprocess the research data into the json file.")

    with open(input_file, 'r', encoding='utf-8') as f:
        f = f.readlines()
        total_num = len(f)
        total_line = 0
        tfrecord_path = os.path.join(tfrecord_dir_path, 'valid.tfrecord')
        with tf.python_io.TFRecordWriter(tfrecord_path) as valid_writer:
            print('Now start writing validing data')
            for eachline in tqdm(f,total=total_num):
                data = json.loads(eachline)
                testid = data['testid']
                features_content = data['features_content']
                labels_index = data['labels_index']
                content_id = _token_to_index(features_content)
                pad_content_id = pad_sequence(content_id, maxlen=pad_seq_len, value=0)
                one_hot_label = create_onehot_labels(labels_index,num_labels)
                example = create_exmple(testid, pad_content_id,one_hot_label)
                valid_writer.write(example.SerializeToString())
                total_line+=1
        print('The number of validing data:{}'.format(total_line))


if __name__=='__main__':
    TRAININGSET_DIR = '../data/train_data.json'    #训练样本所在的文件夹
    VALIDATIONSET_DIR = '../data/valid_data.json'  #验证样本所在文件夹
    data_dir='../data'  #数据需要保存到的文件夹
    pad_seq_len=30      #样本序列最大长度，不足的代码会补0，超过的会截断
    num_labels=4848     #样本类别数
    load_train_data_and_labels(TRAININGSET_DIR,pad_seq_len=pad_seq_len,data_dir=data_dir,num_labels=num_labels)
    load_valid_data_and_labels(VALIDATIONSET_DIR,pad_seq_len=pad_seq_len,data_dir=data_dir,num_labels=num_labels)
