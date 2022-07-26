from itertools import count
import torch
import pandas as pd
import re
import numpy as np
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import pickle
import torch.multiprocessing
#import parmap
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

from listup import *


def feature_extraction(f):
    # open and read the file
    fr = open(f, "r")
    lines = fr.readlines()
    
    # lists that will be return
    x = [] # node features
    edge_attr = [] # edge features
    edge_index = [] # graph connectivity
    y = [] # label

    # lists for constructing each list
    row_index = []
    col_index = []

    # regular expression
    p = re.compile('\"\d+\" \[') # distinguish node
    q = re.compile("\d+") # distinguish node id
    r = re.compile("\"\w+\:[\s\S]*?\"") # edge label "~~"
    s = re.compile("\([\s\S]*?\)\"") # node label
    # for all edge label -> "\[ \w+ = \"\w+\:[\s\S]*?\"\]"

    for line in lines:
        node_feature = [] # init per line
        edge_feature = [] # init per line

        if p.match(line) != None : # node
            # x
            if s.search(line) == None:
                tmp = ''
            else:
                tmp = s.search(line)[0]
                tmp = tmp.replace('\"', '')
            
            node_feature.append(tmp)
            x.append(node_feature)

        elif line.find("->") != -1: # edge
            all_node = q.findall(line)
            if all_node is not None:
                # edge_index -> row & col
                row_index.append(int(all_node[0]))
                col_index.append(int(all_node[1]))
            
            # edge_attr
            if r.search(line) == None:
                print(line)
                break
            tmp = r.search(line)[0]
            tmp = tmp.replace('\"', '')
            edge_feature.append(tmp)
            
            edge_attr.append(edge_feature)

    # edge_index
    #row_index = torch.tensor(row_index)
    #col_index = torch.tensor(col_index)
    edge_index.append(row_index)
    edge_index.append(col_index)
    edge_index = torch.tensor(edge_index)

    # close the file
    fr.close()

    # labeling
    y.append(int(f.split("/")[-2]))
    y = torch.tensor(y)

    return x, edge_attr, edge_index, y


def save_as_pickles(x, edge_attr, edge_index, y, savepath="../temp_data_2"):
    if not os.path.exists("../temp_data_2"):
        os.system("mkdir ../temp_data_2")
    
    # stack in a list then save
    data_ = []
    data_.append(x)
    data_.append(edge_attr)
    data_.append(edge_index)
    data_.append(y)

    torch.save(data_, savepath)

    """
    # save each attribute separately
    xpath = savepath + ".x"
    attrpath = savepath + ".edgeattr"
    edgepath = savepath + ".edgeindex"
    ypath = savepath + ".y"

    torch.save(x, xpath)
    torch.save(edge_attr, attrpath)
    torch.save(edge_index, edgepath)
    torch.save(y, ypath)
    """

    print(savepath)
    
    return


# RegexpTokenizer: tokenize words using regular expression formular
def nltk_tokenizer(_wd):
    return RegexpTokenizer('[^,]+').tokenize(_wd)


def doc2vec(x, edge_attr):
    # epoch to run doc2vec model
    max_epochs = 50

    # save each features to the list
    nfeats = []
    efeats = []

    df_nfeat = pd.DataFrame(x, columns=['feat'])
    df_efeat = pd.DataFrame(edge_attr, columns=['feat'])

    # concat 2 dataframes into 1 dataframe (1 columns)
    df_feat = pd.concat([df_nfeat, df_efeat], axis=0)

    # add index column
    df_feat.index = np.arange(0, len(df_nfeat)+len(df_efeat))
    df_feat.index.name = 'id'
    df_feat = df_feat.reset_index()

    # tokenize
    df_feat['feat'].apply(nltk_tokenizer)

    doc_df = df_feat[['id', 'feat']].values.tolist()
    tagged_data = [TaggedDocument(words=_d, tags=[uid]) for uid, _d in doc_df]

    # build model
    model = Doc2Vec(
        window = 1,
        vector_size = 10,
        alpha = 0.025,
        min_alpha = 0.005,
        min_count = 1,
        dm = 1,
        negative = 5,
        seed = 9999)

    model.build_vocab(tagged_data)

    # train model
    for epoch in range(max_epochs):
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    #model.random.seed(9999)

    nodefeat = []
    edgefeat = []

    cnt = 0

    # save vectorized features
    for i in enumerate(model.dv):
        if cnt < len(x): # for node features
            nodefeat.append(i[1])
        else: # for edge features
            edgefeat.append(i[1])

        cnt += 1

        # To avoid the error 'key not present'
        if cnt > len(tagged_data)-1:
            break

    # list to tensor
    nodefeat = torch.tensor(nodefeat)
    edgefeat = torch.tensor(edgefeat)

    return nodefeat, edgefeat


def generate_data(f):
    # specify file name for saving
    f_ = f.split("complete_add/")[-1]
    f_ = f_.split(".dot")[0] 
    f_ = f_.replace("/", "_")

    savepath = "../temp_data_2/" + f_ + ".pickle"

    # check whether the file is existed or not
    if os.path.exists(savepath):
        return

    # init var
    x = []
    y = []
    edge_index = []
    edge_attr = []

    # extract each list
    x, edge_attr, edge_index, y = feature_extraction(f)
    
    # limit the number of nodes and edges to reduce processing time
    if len(x) + len(edge_attr) > 2000:
        return

    else:
        # get vectorized features
        x, edge_attr = doc2vec(x, edge_attr)

        # convert to tensor
        x = torch.tensor(x)
        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(edge_index)
        y = torch.tensor(y)

        # save as pickles
        save_as_pickles(x, edge_attr, edge_index, y, savepath)

    return
    

def process_dataset(path):
    proclist = listup(path, "d")

    """
    # list for gathering all x, attr, edge, y
    all_x = []
    all_attr = []
    all_index = []
    all_y = []
    

    # stack the features from graph in a list
    def inner_function(x, edge_attr, edge_index, y):
        # gathering
        all_x.append(x)
        all_attr.append(edge_attr)
        all_index.append(edge_index)
        all_y.append(y)
    """

    # multiprocessing
    pool = Pool()
    
    print("PROCESSING START!")
    
    """
    ISSUE:
    I tried to add tqdm or parmap in pool.map for checking the progress
    But, tqdm has tangled code, and parmap is not working (progress bar doesn't update)
    Currently, just stop to implement progress bar.
    """

    """
    # multiprocessing using parmap
    num_cores = mp.cpu_count()
    x, edge_attr, edge_index, y = parmap.starmap_async(generate_data, proclist, pm_pbar=True, pm_processes=num_cores)
    """

    with pool:
        #x, edge_attr, edge_index, y = pool.map(generate_data, proclist)
        pool.map(generate_data, proclist)

        """
        # check whether the list is empty or not
        if not (x and edge_attr and edge_index and y):
            # gathering
            inner_function(x, edge_attr, edge_index, y)
        
        # separately save the graphs
        if not (x and edge_attr and edge_index and y) :
            savepath = "../data/" + f
            if not os.path.exists((savepath +".x")):
                save_as_pickles(x, edge_attr, edge_index, y, savepath)
        """

    print("PROCESSING FINISHED!")

    #torch.multiprocessing.set_sharing_strategy('file_system')

    """
    # convert to tensor
    all_x = torch.tensor(all_x)
    all_attr = torch.tensor(all_attr)
    all_index = torch.tensor(all_index)
    all_y = torch.tensor(all_y)

    # save as file
    print("SAVE INTO PICKLES")
    save_as_pickles(all_x, all_attr, all_index, all_y)
    """

    return