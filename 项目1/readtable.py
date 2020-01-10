import pandas as pd
import numpy as np
import re
import torchtext.vocab as Vocab
import collections

train_df=pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")

#分词
train_list=train_df["text"].values
test_list=test_df['text'].values
labels=list(train_df["target"].values)
print("原始训练数据样本数:",len(train_list))
print("原始测试数据样本数:",len(test_list))
def get_tokenized(data):
    def tokenizer(text):
        text1=[]
        for tok in text.split(' '):
            if len(tok)==0:
                continue
            if tok[0:4] != 'http' and len(re.sub('[^a-zA-Z]', '', tok)) != 0:
                text1.append(re.sub('[^a-zA-Z]', '', tok))
        return [tok.lower() for tok in text1]
    return [tokenizer(review) for review in data]

train_tokenized=get_tokenized(train_list)
test_tokenized=get_tokenized(test_list)
# print(train_tokenized[10])

#去除停用词
stop_word=['i', 'via','me', 'my', 'myself', 'we','us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
           'yourselves','he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
           'they', 'them', 'their','theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
           'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had','having',
           'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
           'while', 'of', 'at', 'by', 'for', 'with', 'about','against', 'between', 'into', 'through', 'during',
           'before', 'after', 'above', 'below', 'to', 'from','up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
           'again', 'further','then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
           'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no','nor', 'not', 'only', 'own', 'same', 'so',
           'than', 'too', 'very', 's', 't', 'can', 'will', 'may','just', 'dont', 'should','could', 'now', 'd', 'll', 'm', 'o', 're',
           've', 'y','aint', 'arent', 'couldnt', 'didnt', 'doesnt', 'hadnt', 'hasnt', 'havent', 'isnt','ma', 'mightnt', 'mustnt',
           'neednt', 'shant', 'shouldnt', 'wasnt', 'werent', 'wont', 'wouldnt']
def delete_stopword(data,stop_word):
    def delete(text):
        text1=[]
        for tok in text:
            if tok not in stop_word:
                text1.append(tok)
        return text1
    return [delete(review) for review in data]
train_tokenized=delete_stopword(train_tokenized,stop_word)
test_tokenized=delete_stopword(test_tokenized,stop_word)
# print(test_tokenized[0])
print("处理后的剩余训练样本数:",len(train_tokenized))
print("处理后的剩余测试样本数:",len(test_tokenized))


#创建词典并删除低频词
copy_train=train_tokenized.copy()
all_list=train_tokenized+test_tokenized

# def get_vocab(data):
#     all_words=sorted([tk for st in data for tk in st])
#     counter=collections.Counter(all_words)
#     #过滤掉词频小于5的词
#     counter=dict(filter(lambda x:x[1]>=5,counter.items()))
#     idx_to_token=[tk for tk,_ in counter.items()]
#     #建立“词---索引”的字典
#     token_to_idx={tk:(idx+1) for idx,tk in enumerate(idx_to_token)}
#     idx_to_token={i:j for j,i in token_to_idx.items()}
#     return token_to_idx,idx_to_token
#获取新的数据
def get_data(data,token_to_idx,max_l):
    dataset=[[token_to_idx[tk] for tk in st if tk in token_to_idx] for st in data]
    def pad(x,max_l):
        return x[:max_l] if len(x) > max_l else x+[0]*(max_l-len(x))
    dataset=[pad(words,max_l) for words in dataset]
    return dataset
# token_to_idx,idx_to_token=get_vocab(all_list)
# train_features=get_data(train_tokenized,token_to_idx,20)
# test_features=get_data(test_tokenized,token_to_idx,20)
# print("索引化后的训练数据:",len(train_features))
# print("索引化后的测试数据:",len(test_features))
# print("训练集标签:",len(labels))

# train_join=[" ".join([tk for tk in st]) for st in train_tokenized]
# test_join=[" ".join([tk for tk in st]) for st in test_tokenized]
import gensim
# model = gensim.models.Word2Vec(all_list, size=100,min_count=5)
# model.wv.save_word2vec_format('./word_vec.txt', binary=False)

import numpy as np
from collections import defaultdict
word_vec_model = gensim.models.KeyedVectors.load_word2vec_format("word_vec.txt", binary=False)
def build_embeddings_matrix(word_vec_model):
    # 初始化词向量矩阵
    embeddings_matrix = np.random.random((len(word_vec_model.vocab)+1, 100))
    # 初始化词索引字典
    word_index = defaultdict(dict)
    for index, word in enumerate(word_vec_model.index2word):
        word_index[word] = index + 1
        # 预留0行给查不到的词
        embeddings_matrix[index+1] = word_vec_model.get_vector(word)
    index_word={i:j for j,i in word_index.items()}
    return word_index, index_word,embeddings_matrix
word_index, index_word,embeddings_matrix=build_embeddings_matrix(word_vec_model)
train_features=get_data(train_tokenized,word_index,20)
test_features=get_data(test_tokenized,word_index,20)

for i,j in zip(train_features,labels):
    if list(set(i))==[0]:
        train_features.remove(i)
        labels.remove(j)
for i in test_features:
    if list(set(i))==[0]:
        test_features.remove(i)
print("索引化后的训练数据:",len(train_features))
print("索引化后的测试数据:",len(test_features))
print("训练集标签:",len(labels))


from model import generator,BiRNN,train,evaluate_accuracy
from torch import nn,optim
import torch
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(np.array(train_features), np.array(labels), test_size=0.2)

x_train=torch.tensor(x_train,dtype=torch.long)
y_train=torch.tensor(y_train)
x_test=torch.tensor(x_test,dtype=torch.long)
y_test=torch.tensor(y_test)

test_features=torch.tensor(np.array(test_features),dtype=torch.long)
embeddings_matrix=torch.tensor(embeddings_matrix)

net=BiRNN(word_index,100,100,2)
net.embedding.weight.data.copy_(embeddings_matrix)
net.embedding.weight.requires_grad=False
loss=nn.CrossEntropyLoss()
optimizer=optim.Adam(filter(lambda p:p.requires_grad,net.parameters()),lr=0.0001)
num_epochs=10
train_iter=generator(x_train,y_train,10,train=True)
test_iter=generator(x_test,y_test,10,train=False)

train(train_iter,test_iter,net,loss,optimizer,num_epochs)