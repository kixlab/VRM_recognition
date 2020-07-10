from gensim.models import KeyedVectors
import torch
import numpy as np
import pdb

class word2vec():
    def __init__(self, model_fn, binary=True, embedding_len=300, sent_len=21):
        embedding_model = KeyedVectors.load_word2vec_format(model_fn, binary="True")
        self.word_vector = embedding_model.wv
        self.embedding_len = embedding_len
        self.sent_len = sent_len
        self.word_vector['.']=np.zeros(300,dtype=np.float32)
        self.word_vector['?']=np.zeros(300,dtype=np.float32)
        self.word_vector['!']=np.zeros(300,dtype=np.float32)

    def to_vector(self,utterance):
        return self.utterance_to_vectors(utterance)


    def utterance_to_vectors(self,utterance):

        def word_to_vector(word):
            try:
                return self.word_vector[word]
            except:
                return np.zeros(self.embedding_len)

        padded_torch = torch.zeros(self.sent_len,self.embedding_len)

        if utterance and len(utterance) !=0 :

            padded_torch[:len(utterance)] = torch.tensor([word_to_vector(word) for word in utterance[:self.sent_len]])


        return padded_torch

