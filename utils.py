from platform import node
import pandas as pd
import numpy as np
import torch

import regex as re

from gensim.models import Word2Vec
from collections import Counter
from sklearn.feature_extraction import DictVectorizer

import sklearn
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler

import networkx as nx
import dgl

def whitespace_clean(text):
    try:
        text = re.sub('[^A-Za-z]', ' ', text)
        text = re.sub(r'\s+', '', text)
        text = text.strip()
        text = text.upper()
    except TypeError:
        pass
    except AttributeError:
        pass
    return text

def processing_bl(cargo, bl_dir, bl_file):
    bl_data = pd.read_csv(bl_dir+bl_file, encoding='cp949', engine='python')
    bl_data = bl_data[['선명','입항년도', '호출부호', '입항횟수','입출항일시','수출입구분','시설명','적하국가명','적하항구명','양하국가명','양하항구명','품목코드','품명','중량톤','용적톤','송하인','수하인']]
    
    # Null processing for throughput
    bl_data = bl_data.loc[~(bl_data['중량톤'].isnull()&bl_data['용적톤'].isnull()),:].reset_index(drop=True)
    bl_data['용적톤'] = bl_data['용적톤'].fillna(0.0)
    bl_data['중량톤'] = bl_data['중량톤'].fillna(0.0)

    # Processing Datetime
    bl_data['date'] = pd.to_datetime(bl_data['입출항일시']).dt.to_period('S')
    bl_data = bl_data.sort_values(by='date', ascending=True).reset_index(drop=True) #시간 순으로 변경

    # Entry Purpose & Country
    bl_data['purpose'] = np.where(bl_data['수출입구분'].isin(['II','IT']), '수입', '수출')
    bl_data['country'] = np.where(bl_data['purpose'].isin(['수입']), bl_data['적하국가명'], bl_data['양하국가명'])

    # Category (Define category from HS code(품목코드))
    bl_data['code1']=bl_data['품목코드'].apply(lambda x: str(x)[:2])
    bl_data['code2']=bl_data['품목코드'].apply(lambda x: str(x)[:4])
    bl_data['category'] = 'NA'
    
        #LBC
    bl_data['category'] = np.where(bl_data['code2'].isin(['2709']), 'crude', bl_data['category'])
    bl_data['category'] = np.where(bl_data['code2'].isin(['2710','2712','2713']), 'oil_products', bl_data['category'])
    bl_data['category'] = np.where(bl_data['code2'].isin(['2705','2711']), 'gas', bl_data['category'])
    bl_data['category'] = np.where(bl_data['code1'].isin(['28','29','30','32','33']), 'chemical', bl_data['category'])
    bl_data['category'] = np.where(bl_data['code1'].isin(['15']), 'plant_animal', bl_data['category'])
    
        # DBC
    bl_data['category'] = np.where(bl_data['code2'].isin(['2701', '2702','2703','2704']), 'coal', bl_data['category'])
    bl_data['category'] = np.where(bl_data['code1'].isin(['10','11']), 'corn', bl_data['category'])
    bl_data['category'] = np.where(bl_data['code1'].isin(['31']), 'fertilizer', bl_data['category'])
    bl_data['category'] = np.where(bl_data['code1'].isin(['26']), 'ore', bl_data['category'])
    bl_data['category'] = np.where(bl_data['code1'].isin(['25']), 'sand', bl_data['category'])
    bl_data['category'] = np.where(bl_data['code1'].isin(['72']), 'steel', bl_data['category'])



    # Change variable name & processing
    bl_data['throughput'] = np.where(bl_data['중량톤']>bl_data['용적톤']*0.883, bl_data['중량톤'], bl_data['용적톤'])
    bl_data['hscode'] = bl_data['품목코드'].astype(str)
    bl_data['name'] = bl_data['품명'].apply(lambda x: whitespace_clean(x))
    bl_data['time'] = bl_data['date'].apply(lambda x: str(x)[:7])
    
    bl_data = bl_data[['time','country','purpose','category','hscode', 'name','throughput']]
    lbc_cat = ['crude','oil_products','gas','chemical','plant_animal']
    dbc_cat = ['coal','corn','fertilizer','ore','sand','steel']
    
    if cargo = 'LBC':
        lbc_data = bl_data[bl_data['category'].isin(lbc_cat)].reset_index(drop=True)
        lbc_data['flag'] = lbc_data.apply(lambda x: x['time']+'-'+x['country'], axis=1)
        
        # tt_data
        pivot = pd.DataFrame([t+'-'+v for t in lbc_data['time'].unique() for v in lbc_data['country'].unique()], columns = ['flag'])
        im_core = lbc_data[lbc_data['purpose'] == '수입'].groupby('flag').agg({'category':lambda x: list(x), 'hscode':lambda x: list(x), 'name':lambda x: list(x)}).reset_index()
        ex_core = lbc_data[lbc_data['purpose'] == '수출'].groupby('flag').agg({'category':lambda x: list(x), 'hscode':lambda x: list(x), 'name':lambda x: list(x)}).reset_index()   

        tt_im_data = pd.merge(pivot, im_core, how = 'left', on='flag')
        tt_im_data = tt_im_data.apply(lambda x: x.fillna({i: [] for i in tt_im_data.index}))
        
        tt_ex_data = pd.merge(pivot, ex_core, how = 'left', on='flag')
        tt_ex_data = tt_ex_data.apply(lambda x: x.fillna({i: [] for i in tt_ex_data.index}))
        
        # node_feature
        node_feature = lbc_data.pivot_table(index = ['time','country'], columns='category', values = 'throughput', aggfunc='sum')[lbc_cat].reset_index()
        node_feature['flag'] = node_feature.apply(lambda x: x['time']+'-'+x['country'], axis=1)
        node_feature = pd.merge(pivot, node_feature, how = 'left', on = 'flag')
        node_feature = node_feature.fillna(0)
        node_feature = node_feature.iloc[:,3:].values.reshape(len(lbc_data['time'].unique()),len(lbc_data['country'].unique()),-1)
        
        cat_list = lbc_cat
        
    elif cargo = 'DBC':
        dbc_data =  bl_data[bl_data['category'].isin(dbc_cat)].reset_index(drop=True)
        dbc_data['flag'] = dbc_data.apply(lambda x: x['time']+'-'+x['country'], axis=1)
        
        # tt_data
        pivot = pd.DataFrame([t+'-'+v for t in dbc_data['time'].unique() for v in dbc_data['country'].unique()], columns = ['flag'])
        im_core = dbc_data[dbc_data['purpose'] == '수입'].groupby('flag').agg({'category':lambda x: list(x), 'hscode':lambda x: list(x), 'name':lambda x: list(x)}).reset_index()
        ex_core = dbc_data[dbc_data['purpose'] == '수출'].groupby('flag').agg({'category':lambda x: list(x), 'hscode':lambda x: list(x), 'name':lambda x: list(x)}).reset_index()   

        tt_im_data = pd.merge(pivot, im_core, how = 'left', on='flag')
        tt_im_data = tt_im_data.apply(lambda x: x.fillna({i: [] for i in tt_im_data.index}))
        
        tt_ex_data = pd.merge(pivot, ex_core, how = 'left', on='flag')
        tt_ex_data = tt_ex_data.apply(lambda x: x.fillna({i: [] for i in tt_ex_data.index}))
        
        # node_feature
        node_feature = dbc_data.pivot_table(index = ['time','country'], columns='category', values = 'throughput', aggfunc='sum')[dbc_cat].reset_index()
        node_feature['flag'] = node_feature.apply(lambda x: x['time']+'-'+x['country'], axis=1)
        node_feature = pd.merge(pivot, node_feature, how = 'left', on = 'flag')
        node_feature = node_feature.fillna(0)
        node_feature = node_feature.iloc[:,3:].values.reshape(len(dbc_data['time'].unique()),len(dbc_data['country'].unique()),-1)
        
        cat_list = dbc_cat
    
    return tt_im_data, tt_ex_data, node_feature, cat_list

def generate_ttd_corpus(data, purpose, level):
    document = []
    for i in range(len(data)):
#         document.append(list(filter(None,data.iloc[i,level].replace("[", "").replace("]", "").replace('""', '').replace("'", "").replace(" ", "").split(","))))
        document.append(data.iloc[i,level])
    return document

def word2vec(document, window, size, wv_dir, purpose, level):
    model = Word2Vec(document, min_count=1, window= window, size = size, sg =1, iter = 100)
    model.init_sims(replace=True)
    model.wv.save_word2vec_format(wv_dir+'p{}_l{}_w2v.txt'.format(purpose,level)) #

def generate_ttd_embedding(data, purpose, level, args):
    document = generate_ttd_corpus(data, purpose, level)
    word2vec(document, args.window, args.size, args.wv_dir, purpose, level)

    corpus = []
    for i in range(len(document)):
        corpus.append(dict(Counter(document[i])))
    v = DictVectorizer(sparse=False)
    dtm = v.fit_transform(corpus)

    w2v = pd.read_csv(args.wv_dir+'p{}_l{}_w2v.txt'.format(purpose,level), sep = ' ', engine = 'python', encoding = 'utf-8')
    w2v = w2v.sort_index(ascending=True)
    w2v.reset_index(inplace=True)  
    w2v = w2v.drop(['level_0'], axis = 1)

    ttd_embedding = np.inner(dtm, np.array(w2v).T)
    ttd_embedding = pd.DataFrame(ttd_embedding)
    ttd_embedding.index  = data.iloc[:,0]
    return ttd_embedding
    

def sim_matrix(c_ttd_embedding, time, cut_off):
    unique_month = c_ttd_embedding['month'].unique().tolist()
    
    vec = c_ttd_embedding[c_ttd_embedding['month'] == unique_month[time]].iloc[:,1:-2]
    
    #similarity matrix
    cosine_sim = pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(vec, Y=None, dense_output=True))
    norm_sim = (cosine_sim + 1) / 2
    norm_sim[norm_sim<cut_off]=0
    norm_sim[norm_sim>=cut_off]=1
    norm_sim = norm_sim - np.diag(np.diag(norm_sim))
    #norm_sim = cosine_sim - np.diag(np.diag(cosine_sim))
    return norm_sim

def all_nx_graph(c_ttd_embedding, time, cut_off):
    norm_sim = sim_matrix(c_ttd_embedding, time, cut_off)    
    node_country = c_ttd_embedding['country'].values.tolist()
    H = nx.from_pandas_adjacency(norm_sim)
    return H

def build_dataset_y(y, seq_length):
    dataY=[]
    for i in range(0, len(y)-seq_length):
        _y=y[i+seq_length]
        dataY.append(_y)
    return np.array(dataY)

def build_dataset(table, seq_length):
    tensor=[]
    for i in range(0, len(table)-seq_length):
        _x=table[i:i+seq_length,:]
        tensor.append(_x)
    return torch.stack(tensor)   

def train_test_data(node_feature, c_ttd_embedding, split_ratio, cat_list, args):
    '''
    node_feature: (#time, #country, #category)
    '''
    
    train_size = int(node_feature.shape[0]*split_ratio)
    scaler = (node_feature[:train_size]).reshape(-1,node_feature.shape[2])
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(scaler)
    
    G_list = [] # [G_1,G_2,...,G_T]
    
    for t in range(node_feature.shape[0]):
        g = all_nx_graph(c_ttd_embedding, t, args.cut_off)
        g = dgl.from_networkx(g)
        g.ndata['x'] = torch.tensor(minmax_scaler.transform(node_feature[i]),dtype = torch.float32)
        G_list.append(g)

    train_G_list = G_list[0:train_size]
    test_G_list =  G_list[train_size-args.seq_length:]

    # ITSG generation
    train_G = dgl.batch(train_G_list)
    test_G = dgl.batch(test_G_list)

    # Throughput statistics
    statistics = node_feature.sum(axis = 1) # (#time,  #category) 
    total_throughput = statistics.sum(axis=1).reshape(-1,1)
    statistics = np.hstack((statistics,total_throughput)) # (#time,  (#category + 1))

    # Target data
    if args.target = 'total':
        trainY = statistics[0:train_size,-1].reshape(-1,1)
        testY = statistics[train_size-args.seq_length:,-1].reshape(-1,1)
    else:
        trainY = statistics[0:train_size,cat_list.index(args.target)].reshape(-1,1)
        testY = statistics[train_size-args.seq_length:,cat_list.index(args.target)].reshape(-1,1)
    minmax_scaler_y = MinMaxScaler()
    minmax_scaler_y.fit(trainY)
    trainY = minmax_scaler_y.transform(trainY)
    testY = minmax_scaler_y.transform(testY)

    trainY =  build_dataset_y(trainY, seq_length)
    testY =  build_dataset_y(testY, seq_length)
    trainY_tensor=torch.FloatTensor(trainY)
    testY_tensor=torch.FloatTensor(testY)
    trainY=Variable(trainY_tensor)
    testY=Variable(testY_tensor)
    return train_G, test_G, trainY, testY, minmax_scaler, minmax_scaler_y, G_list


