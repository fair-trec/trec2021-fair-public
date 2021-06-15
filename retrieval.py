#!/usr/bin/env python3

import os
from sys import meta_path
import pyterrier as pt 
import json
import pandas as pd 
import time
import re

pt.init()


corpus_path = "path/to/trec_corpus.json.gz"
topics_file = "path/to/trec_topics.json"
index_path = "./corpus_idx"


def stream_gzipped_lines(fileObject):
    with fileObject as gf:
        for ln in gf:
            yield ln

def stream_gzipped_json_list(fileObject):
    for ln in stream_gzipped_lines(fileObject):
        if ln == b'[\n' or ln == b']\n':
            continue
        yield json.loads(ln)

def fairJsonReturn(fileOpen):
    start = time.monotonic()
    for obj in stream_gzipped_json_list(fileOpen):
        print(obj['id'], end=' ')
        yield obj
    end = time.monotonic()
    print("fairJsonReturn took so long: {}".format(end - start))
        
     
def create_index(index_path):
    #Create Corpus index with multiple threads
    start = time.monotonic()
    iter_index_Corpus = pt.IterDictIndexer(index_path, threads=1)
    end = time.monotonic()
    print("pt.IterDictIndexer creation took: {}".format(end - start))
    return iter_index_Corpus

def get_topics_qrels(topics_file):
    topics_data = []
    qrel_data = []
    with open(topics_file, 'rt') as f:
        for line in f:
            t = json.loads(line)
            topics_data.append( [ str(t['id']), " ".join( [ re.sub(r"'.*", "", kw) for kw in t['keywords'] ]  ) ] )
            for doc in t['rel_docs']:
                qrel_data.append( [ str(t['id']), str(doc), 1] )
    return topics_data, qrel_data


corpus = pt.io.autoopen(corpus_path, mode='rt')
iter_index_Corpus = create_index(index_path)

if os.path.exists(index_path):
    print("Index already exists!")
else:
    start = time.monotonic()
    indexRef_Corpus = iter_index_Corpus.index(fairJsonReturn(corpus))
    end = time.monotonic()
    print("pt.IterDictIndexer.index took so long: {}".format(end - start))


index = pt.IndexFactory.of(indexRef_Corpus)

topics_data, qrel_data = get_topics_qrels(topics_file)

topics = pd.DataFrame(topics_data, columns=["qid", "query"])
qrels = pd.DataFrame(qrel_data, columns=["qid", "docno", "label"])

tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
bm25 = pt.BatchRetrieve(index, wmodel="BM25")
pl2 = pt.BatchRetrieve(index, wmodel="PL2")

res = pt.Experiment([tf_idf, bm25, pl2], topics, qrels, eval_metrics=["map"])

print(res)