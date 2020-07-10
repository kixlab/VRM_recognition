import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pdb

S = 1 # speaker
O = 0 # other
ignore_index= -1

label_map_SE = {'A':S,
                'D':S,
                'E':S,
                'K':O,
                'Q':O,
                'I':O,
                'R':O,
                'C':S
                }

label_map_PE = {'A':O,
                'D':S,
                'E':S,
                'K':S,
                'Q':S,
                'I':O,
                'R':O,
                'C':O
                }

label_map_FR = {'A':S,
                'D':S,
                'E':O,
                'K':O,
                'Q':S,
                'I':S,
                'R':O,
                'C':O
                }

pos_map = {'LS': 0, 'TO': 1, 'VBN': 2, "''": 3, 'WP': 4, 'UH': 5, 'VBG': 6, 'JJ': 7, 'VBZ': 8, '--': 9,
    'VBP': 10, 'NN': 11, 'DT': 12, 'PRP': 13, ':': 14, 'WP$': 15, 'NNPS': 16, 'PRP$': 17, 'WDT': 18,
    '(': 19, ')': 20, '.': 21, ',': 22, '``': 23, '$': 24, 'RB': 25, 'RBR': 26, 'RBS': 27, 'VBD': 28,
    'IN': 29, 'FW': 30, 'RP': 31, 'JJR': 32, 'JJS': 33, 'PDT': 34, 'MD': 35, 'VB': 36, 'WRB': 37,
    'NNP': 38, 'EX': 39, 'NNS': 40, 'SYM': 41, 'CC': 42, 'CD': 43, 'POS': 44, 'START':45, 'END':46}


class VRMDataset(Dataset):

    def __init__(self,data_df, to_vector, label_mapper,sent_len,pos_tag_len, max_dialogue_len,chunk_size=1,
                use_setoken=False,by_conversation=False,lastpooling=False):
        self.df = data_df
        self.use_setoken = use_setoken
        self.sent_len = sent_len
        self.pos_tag_len = pos_tag_len
        self.lastpooling=lastpooling

        self.max_dialogue_len= max_dialogue_len


        utterances_pos_tagged = self.df['utterance'].apply(self._pos_tagged_tokenizer_VRM)
        self.df['utterance'] = [token[0] for token in utterances_pos_tagged]
        self.df['pos'] = [token[1] for token in utterances_pos_tagged]

        self.by_conversation = by_conversation

        if by_conversation:
            self.conversation_ids = list(set(self.df['conversation_id']))
            self.conversation_ids.sort()

            self.conversation_scopes = []
            i = 0
            start = 0

            for end, conversation_id in enumerate(self.df['conversation_id']):

                if conversation_id == i:
                    continue
                else:
                    i = conversation_id
                    self.conversation_scopes.append([start,end])
                    start = end
            self.conversation_scopes.append([start,end])
            self.index_list = [i for i in range(len(self.conversation_scopes))]
        else :
            self.conversation_ids = list(self.df['conversation_id'])
            self.conversation_scopes = []
            self.index_list = [i for i in range(len(self.df['utterance']))]

        self.to_vector = to_vector
        self.label_mapper = label_mapper
        self.chunk_size = chunk_size


    def __len__(self):
        return len(self.index_list)

    def _get_single_utterance(self,idx,conversation_id):

        if self.by_conversation:
            utterance = self.df['utterance'][idx]
        else:
            if idx < self.__len__() and conversation_id == self.conversation_ids[idx]:
                utterance = self.df['utterance'][idx]
            else :
                utterance = None

        utter_vec = self.to_vector(utterance)

        return utter_vec

    def _get_single_pos(self,idx,conversation_id):
        if self.by_conversation:
            pos = self.df['pos'][idx]
            pos_vec = self._pos_to_vector(pos)
        else:
            if idx < self.__len__() and conversation_id == self.conversation_ids[idx]:
                pos = self.df['pos'][idx]
                pos_vec = self._pos_to_vector(pos)
            else:
                pos_vec = self._pos_to_vector([])

        return pos_vec

    def _get_single_label(self,idx,conversation_id):
        if self.by_conversation:
            label = self.df['mode'][idx]
        else:
            if idx < self.__len__() and conversation_id == self.conversation_ids[idx]:
                label = self.df['mode'][idx]
            else :
                label = None

        label = self.label_mapper(label)

        return label

    def _get_single_qtag(self,idx,conversation_id):
        if self.by_conversation:
            qtag = 1 if self.df['utterance'][idx][-1]=='?' else 0
        else:
            if idx < self.__len__() and conversation_id == self.conversation_ids[idx]:
                qtag = 1 if self.df['utterance'][idx][-1]=='?' else 0
            else :
                qtag = 0
        return qtag

    def _get_single_mask(self,idx,conversation_id):
        if self.by_conversation:
            if self.lastpooling:
                mask = [0 for _ in range(self.sent_len)]
                mask[min(len(self.df['utterance'][idx])-1,self.sent_len-1)] = 1
            else:
                mask = [1 for _ in range(len(self.df['utterance'][idx]))]
                mask += [0 for _ in range(self.sent_len-len(mask))]
        else:
            if idx < self.__len__() and conversation_id == self.conversation_ids[idx]:
                if self.lastpooling:
                    mask = [0 for _ in range(self.sent_len)]
                    mask[min(len(self.df['utterance'][idx])-1,self.sent_len-1)] = 1
                else:
                    mask = [1 for _ in range(len(self.df['utterance'][idx]))]
                    mask += [0 for _ in range(self.sent_len-len(mask))]
            else :
                mask = [0 for _ in range(self.sent_len)]
        return mask[:self.sent_len]

    def _get_caller(self,start,end,conversation_id):

        callers = []
        if self.by_conversation:
            before_caller = 'None'
            for i in range(start,end):
                if self.df['caller'][i] == before_caller:
                    callers.append(1)
                else:
                    callers.append(0)
                before_caller = self.df['caller'][i]
        else:
            before_caller = 'None'
            for i in range(start,end):
                if i < self.__len__() and conversation_id == self.conversation_ids[i]:
                    if self.df['caller'][i] == before_caller:
                        callers.append(1)
                    else:
                        callers.append(0)
                    before_caller = self.df['caller'][i]
                else :
                    callers.append(0)
        return callers

    def __getitem__(self, idx_0):

        conversation_id = self.conversation_ids[idx_0]
        idx = int(self.index_list[idx_0])
        start = idx
        end = idx + self.chunk_size

        if self.by_conversation:
            idx = idx_0

            start, end = self.conversation_scopes[conversation_id]

            utterance_chunk = [self._get_single_utterance(i,conversation_id) for i in range(start,end)]
            utts_size = len(utterance_chunk)
            utt_shape = utterance_chunk[0].shape
            utterance_chunk = utterance_chunk[:self.max_dialogue_len] + [torch.zeros(utt_shape) for _ in range(self.max_dialogue_len-utts_size)]
            utterance_chunk = torch.stack(utterance_chunk)

            label_chunk = [self._get_single_label(i,conversation_id) for i in range(start,end)]
            label_chunk = label_chunk[:self.max_dialogue_len] + [ self.label_mapper(None) for _ in range(self.max_dialogue_len-utts_size)]
            label_chunk = torch.tensor(label_chunk)

            mask_chunk = [self._get_single_mask(i,conversation_id) for i in range(start,end)]
            mask_chunk = mask_chunk[:self.max_dialogue_len] + [[0 for _ in range(self.sent_len)] for _ in range(self.max_dialogue_len-utts_size)]
            mask_chunk = torch.tensor(mask_chunk)

            pos_chunk = [self._get_single_pos(i,conversation_id) for i in range(start,end)]
            pos_chunk =pos_chunk[:self.max_dialogue_len] + [np.zeros((self.sent_len,self.pos_tag_len)) for _ in range(self.max_dialogue_len-utts_size)]
            pos_chunk = torch.tensor(pos_chunk)

            qtag_chunk = [self._get_single_qtag(i,conversation_id) for i in range(start,end)]
            qtag_chunk = qtag_chunk[:self.max_dialogue_len] + [ 0 for _ in range(self.max_dialogue_len-utts_size)]
            qtag_chunk = torch.tensor(qtag_chunk)

            caller_chunk = self._get_caller(start,end,conversation_id)
            caller_chunk = caller_chunk[:self.max_dialogue_len] + [ 0 for _ in range(self.max_dialogue_len-utts_size)]
            caller_chunk = torch.tensor(caller_chunk)

            return utterance_chunk,pos_chunk, qtag_chunk,caller_chunk, mask_chunk, label_chunk

        else:

            utterance_chunk = torch.stack([self._get_single_utterance(i,conversation_id) for i in range(start,end)])
            label_chunk = torch.tensor([self._get_single_label(i,conversation_id) for i in range(start,end)])
            pos_chunk = torch.tensor([self._get_single_pos(i,conversation_id) for i in range(start,end)])
            mask_chunk = torch.tensor([self._get_single_mask(i,conversation_id) for i in range(start,end)])
            qtag_chunk = torch.tensor([self._get_single_qtag(i,conversation_id) for i in range(start,end)])
            caller_chunk = torch.tensor(self._get_caller(start,end,conversation_id))

            return utterance_chunk,pos_chunk, qtag_chunk,caller_chunk, mask_chunk, label_chunk

    def _get_single_qtag(self,idx,conversation_id):
        if self.by_conversation:
            utt = self.df['utterance'][idx]
            qtag = 1 if len(utt) != 0 and utt[-1]=='?' else 0
        else:
            if idx < self.__len__()  and conversation_id == self.conversation_ids[idx]:
                utt = self.df['utterance'][idx]
                qtag = 1 if len(utt) != 0 and utt[-1]=='?' else 0
            else :
                qtag = 0

        return qtag


    def _pos_tagged_tokenizer_VRM(self,utterance):
        # using pos
        if type(utterance) is str:
            utterance_pos = [word.split('/') for word in utterance.strip().split(' ')]


            utterance = [token[0] for token in utterance_pos]

            pos = [token[1]for token in utterance_pos]

            if self.use_setoken:
                utterance = ['<SOS>'] + utterance + ['<EOS>']
                pos = ['START'] + pos + ['END']
            return utterance, pos
        else :
            return [], []


    def _pos_to_vector(self,pos):
        pos_index = [pos_map[p] for p in pos]

        pos_vector = np.zeros((self.sent_len,self.pos_tag_len))
        pos_index = pos_index[:self.sent_len]

        for i,j in enumerate(pos_index):
            pos_vector[i][j]=1

        return pos_vector

def label_mapper_VRM(label):
    return [label_mapper_VRM_SE_f(label),
            label_mapper_VRM_PE_f(label),
            label_mapper_VRM_FR_f(label),
            label_mapper_VRM_SE(label),
            label_mapper_VRM_PE(label),
            label_mapper_VRM_FR(label)]

def label_mapper_VRM_SE(label):

    if label:
        annot_list = label.split('/')
        # just use first annotation in here and intent
        target = annot_list[0][1]
        if target == 'Z' or target == 'U':
            return ignore_index
        else:

            return label_map_SE[target]

        return
    else:
        return ignore_index

def label_mapper_VRM_PE(label):

    if label:
        annot_list = label.split('/')
        # just use first annotation in here and intent
        target = annot_list[0][1]
        if target == 'Z' or target == 'U':
            return ignore_index
        else:

            return label_map_PE[target]

        return
    else:
        return ignore_index

def label_mapper_VRM_FR(label):

    if label:
        annot_list = label.split('/')
        # just use first annotation in here and intent
        target = annot_list[0][1]
        if target == 'Z' or target == 'U':
            return ignore_index
        else:

            return label_map_FR[target]

        return
    else:
        return ignore_index


def label_mapper_VRM_SE_f(label):

    if label:
        annot_list = label.split('/')
        # just use first annotation in here and intent
        target = annot_list[0][0]
        if target == 'Z' or target == 'U':
            return ignore_index
        else:

            return label_map_SE[target]

        return
    else:
        return ignore_index

def label_mapper_VRM_PE_f(label):

    if label:
        annot_list = label.split('/')
        # just use first annotation in here and intent
        target = annot_list[0][0]
        if target == 'Z' or target == 'U':
            return ignore_index
        else:

            return label_map_PE[target]

        return
    else:
        return ignore_index

def label_mapper_VRM_FR_f(label):

    if label:
        annot_list = label.split('/')
        # just use first annotation in here and intent
        target = annot_list[0][0]
        if target == 'Z' or target == 'U':
            return ignore_index
        else:

            return label_map_FR[target]

        return
    else:
        return ignore_index
