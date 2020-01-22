import pickle
from collections import OrderedDict
import numpy as np
import jieba
import json
from tqdm import tqdm
import tokenization
import os
import pkuseg       #分词工具
seg=pkuseg.pkuseg()




def dump_zhidao(zhidao_path,save_path):
    """
    提取文件中的内容和标签
    :param zhidao_path:
    :param save_path:
    :return:
    """
    path_list=os.listdir(zhidao_path)
    all_data=[]
    for path in path_list:
        path=os.path.join(zhidao_path,path)
        f=open(path,'r',encoding='utf-8')
        for i in f:
            data_dict={}
            data=json.loads(i)
            data_dict["question"]=data['question']
            data_dict["tags"]=data["tags"]
            all_data.append(data_dict)
        f.close()
    with open(save_path,'wb') as f:
        pickle.dump(all_data,f)

def check_tags_distribution(data_path):
    """
    查看标签的分布情况
    :param data_path:
    :return:
    """
    tags_dict={}
    with open(data_path,'rb') as f:
        data=pickle.load(f)
        for i in data:
            for j in i['tags']:
                tags_dict[j]=tags_dict.get(j,0)+1
    tags_dict=sorted(tags_dict.items(),key=lambda x:x[1],reverse=True)
    print(tags_dict)
    print(len(tags_dict))
    print(sum([i[1] for i in tags_dict])/len(tags_dict))



def topic_number(path,tags_min_num=10):
    """
    过滤，只取样本数量大于10的标签
    :param path:
    :return:
    """
    all_data=pickle.load(open(path,'rb'))
    tags_dict={}
    for i in all_data:
        for j in i['tags']:
            tags_dict[j]=tags_dict.get(j,0)+1
    tags_dict={k:v for k,v in tags_dict.items() if v>=tags_min_num}
    # print(tags_dict)
    print(len(tags_dict))
    return tags_dict


def create_lookup_tables(tags_dict):
    """
    构建tags->index，index->tags的字典
    :param tags_dict:
    :return:
    """
    words = sorted(list(set(tags_dict)))
    vocab_to_int = {value:i for i,value in enumerate(words)}
    int_to_vocab = dict(enumerate(words))
    return vocab_to_int, int_to_vocab

def load_stop_words(stopwords_path):
    """
    加载停止词
    :param stopwords_path:
    :return:
    """
    with open(stopwords_path,'r',encoding='utf-8') as f:
        stop_words=list(map(lambda x:x.strip(),f.readlines()))
        # print(stop_words)
    return stop_words

def remove_stop_words(stop_words,content_list):
    """
    过滤句子中的停止词
    :param stop_words:
    :param content_list:
    :return:
    """
    content=[]
    for word in content_list:
        if word.strip() not in stop_words:
            if word.strip()!='':
                content.append(word.strip())
    return content

def create_data(data_path,stop_words,tags_dict,vocab_to_int,is_training,num_threshold=50000):
    """
    生成样本集
    :param data_path:
    :param stop_words:
    :param tags_dict:
    :param vocab_to_int:
    :param is_training:
    :param num_threshold: 限制样本个数，防止样本不均衡
    :return:
    """
    all_data = pickle.load(open(data_path, 'rb'))
    all_tags=set(tags_dict)
    tags_dict={}
    features_id=1
    if is_training:
        f=open('../data/train_data.json','w',encoding='utf-8')
    else:
        f=open('../data/valid_data.json','w',encoding='utf-8')
    for i in tqdm(all_data):
        features_dict = {}
        max_num = max([tags_dict.get(k, 0) for k in i['tags']])
        if set(i['tags']) <= all_tags and max_num < num_threshold:
            features_dict["testid"] = features_id
            content_list = seg.cut(i['question'])
            content = remove_stop_words(stop_words, content_list)
            features_dict["features_content"] = content
            features_dict["labels_index"] = [vocab_to_int[tag] for tag in i['tags'] if tag in vocab_to_int]
            features_dict["labels_num"] = len(i['tags'])
            if content and len(i['tags'])>0:
                f.write(json.dumps(features_dict, ensure_ascii=False) + '\n')
                features_id += 1
    f.close()


def create_charvector_data(data_path,tags_dict,vocab_to_int,num_threshold=100):
    """
    创建字向量
    :param data_path:
    :param tags_dict:
    :param vocab_to_int:
    :param num_threshold:
    :return:
    """
    tokenizer = tokenization.FullTokenizer(
        vocab_file=1, do_lower_case=True)
    all_data = pickle.load(open(data_path, 'rb'))
    all_tags=set(tags_dict)
    tags_dict={}
    features_id=1
    with open('char_sample.json','w',encoding='utf-8') as f:
        for i in tqdm(all_data):
            features_dict = {}
            max_num=max([tags_dict.get(k,0) for k in i['tags']])
            if set(i['tags'])<=all_tags and max_num<num_threshold:
                features_dict["testid"]=features_id
                # content_list=list(i['question'])
                content_list=tokenizer.tokenize(i['question'])
                # print(content_list)
                # break
                features_dict["features_content"]=content_list
                features_dict["labels_index"]=[vocab_to_int[tag] for tag in i['tags']]
                features_dict["labels_num"]=len(i['tags'])
                # data=json.dumps(features_dict)
                # f.write(str(features_dict))
                f.write(json.dumps(features_dict,ensure_ascii=False)+'\n')
                for j in i['tags']:
                    tags_dict[j]=tags_dict.get(j,0)+1
                features_id+=1


def split_char_emdedding(path,save_path,vob_path):
    """
    加载并保存预训练的字向量文件
    :param path:
    :param save_path:
    :param vob_path:
    :return:
    """
    with open(path,'r',encoding='utf-8') as f:
        # with open(save_path,'w',encoding='utf-8') as sp:
            embedding_data=f.readlines()
            vob_2_int={}
            vector_list=[]
            index=0
            for i in range(len(embedding_data)):
                data=embedding_data[i].strip().split(' ')

                if len(data)==301 and len(data[0])==1:
                    if data[0] not in vob_2_int:
                        vob_2_int[data[0]]=index
                        vector_list.append(data[1:301])
                        index+=1
            np.save(save_path,np.array(vector_list,np.float32))
            np.save(vob_path,vob_2_int)

def split_word_emdedding(path,save_path,vob_path):
    """
    加载并保存预训练的词向量文件
    :param path:
    :param save_path:
    :param vob_path:
    :return:
    """
    with open(path,'r',encoding='utf-8') as f:
        # with open(save_path,'w',encoding='utf-8') as sp:
            embedding_data=f.readlines()
            vob_2_int={}
            vector_list=[]
            index=0
            for i in tqdm(range(len(embedding_data))):
                data=embedding_data[i].strip().split(' ')
                if len(data)==301:
                    if data[0] not in vob_2_int:
                        vob_2_int[data[0]]=index
                        vector_list.append(data[1:301])
                        index+=1
            np.save(save_path,np.array(vector_list,np.float32))
            np.save(vob_path,vob_2_int)

def restore_prediction(true_path,prediction_path,int_to_vocab):
    """
    对预测的数据进行还原
    :param true_path:
    :param prediction_path:
    :param int_to_vocab:
    :return:
    """
    id_2_content = {}
    with open(true_path, 'r', encoding='utf-8') as f:
        true_data = f.readlines()
        for i in true_data:
            data = json.loads(i)
            id_2_content[data['testid']] = ''.join(data['features_content'])
    with open(prediction_path, 'r') as f:
        with open('results.json', 'w', encoding='utf-8') as res:
            prediction_data = f.readlines()
            for i in prediction_data:
                result = {}
                data = json.loads(i)
                result['content'] = id_2_content[data['id']]
                result['labels'] = [int_to_vocab[i] for i in data['labels']]
                result['predict_labels'] = [int_to_vocab[i] for i in data['predict_labels']]
                res.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__=='__main__':
    # 数据文件
    train_path='D:/work/多标签分类/train_data'
    valid_path='D:/work/多标签分类/valid_data'

    #词向量文件,网址  https://github.com/Embedding/Chinese-Word-Vectors
    char_path='C:/Users/王本强/Desktop/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5'

    #停止词
    stopwords_path = '../data/stopwords.dat'

    #保存训练文件，拿取问题和标签
    train_data='../data/train.pkl'
    valid_data='../data/valid.pkl'

    dump_zhidao(train_path,train_data)
    dump_zhidao(valid_path,valid_data)

    #保存词向量文件
    save_path='../data/word_vec/word_vector'
    vob_path='../data/word_vec/vob_2_int'
    split_word_emdedding(char_path,save_path,vob_path)


    #词向量训练数据
    tags_dict = topic_number(train_data)
    vocab_to_int, int_to_vocab = create_lookup_tables(tags_dict)
    stop_words=load_stop_words(stopwords_path)
    create_data(train_data,stop_words,tags_dict,True,vocab_to_int)
    create_data(valid_data,stop_words,tags_dict,False,vocab_to_int)




    #查看标签个数分布
    # check_tags_distribution('../data/zhidao.pkl')


    # 保存字向量文件
    # char_path='C:/Users\王本强\Desktop/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5'
    # save_path='C:/Users\王本强\Desktop\Multi-Label-Text-Classification-word\data\word_vec/word_vector'
    # vob_path='C:/Users\王本强\Desktop\Multi-Label-Text-Classification-word\data\word_vec/vob_2_int'
    # split_char_emdedding(char_path,save_path,vob_path)

    #字向量训练数据
    # tags_dict=topic_number(data_path)
    # print(len(tags_dict))
    # vocab_to_int,int_to_vocab=create_lookup_tables(tags_dict)
    # create_charvector_data(data_path,tags_dict,vocab_to_int)

    #还原预测数据
    # tags_dict = topic_number(data_path)
    # vocab_to_int, int_to_vocab = create_lookup_tables(tags_dict)
    # prediction_path='C:/Users\王本强\Desktop\Multi-Label-Text-Classification\data\predictions.json'
    # true_path='C:/Users\王本强\Desktop\Multi-Label-Text-Classification\data/valid_data.json'
    # restore_prediction(true_path,prediction_path,int_to_vocab)

