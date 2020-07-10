import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiLSTM_Enc_RAM(nn.Module):

    def __init__(self,device,max_length,num_tag,embedding_dim,pos_dim, hidden_dim_word,hidden_dim_pos,hidden_dim_att,num_layers=1,dropout_rate = 0.0):

        super(BiLSTM_Enc_RAM,self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.hidden_dim_att = hidden_dim_att
        self.hidden_dim_word = hidden_dim_word
        self.hidden_dim_pos = hidden_dim_pos

        self.input_layer_pos = nn.Linear(pos_dim, hidden_dim_pos*2)
        self.lstm_pos = nn.LSTM(hidden_dim_pos*2, hidden_dim_pos // 2, batch_first = True,
                            num_layers=num_layers, dropout=0,bidirectional=True)



        self.input_layer_embeds = nn.Linear(embedding_dim,hidden_dim_word*2)
        self.lstm = nn.LSTM(hidden_dim_word*2, hidden_dim_word // 2, batch_first = True,
                            num_layers=num_layers, dropout=0,bidirectional=True)


        self.input_layer_att = nn.Linear(hidden_dim_word+hidden_dim_pos,hidden_dim_att)
        self.lstm_att = nn.LSTM(hidden_dim_att, hidden_dim_att // 2, batch_first = True,
                                            num_layers=1, dropout=0,bidirectional=True)
        self.hidden2att = nn.Linear(hidden_dim_att,1)


        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

        # not used. dummy for load pretrained
        self.hidden2tag = nn.Linear(hidden_dim_word+hidden_dim_pos, num_tag)
        self.att =nn.Conv1d(max_length,max_length, hidden_dim_word+hidden_dim_pos,groups=max_length)
        self.fc = nn.Linear(hidden_dim_word+hidden_dim_pos,hidden_dim_word+hidden_dim_pos)

    def init_hidden_pos(self,batch_size):
        return (torch.zeros(2*self.num_layers, batch_size, self.hidden_dim_pos // 2).to(self.device),
                torch.zeros(2*self.num_layers, batch_size,self.hidden_dim_pos // 2).to(self.device))

    def init_hidden_att(self,batch_size):
        return (torch.zeros(2, batch_size, self.hidden_dim_att // 2).to(self.device),
                torch.zeros(2, batch_size,self.hidden_dim_att // 2).to(self.device))

    def init_hidden(self,batch_size):
        return (torch.zeros(2*self.num_layers, batch_size, self.hidden_dim_word // 2).to(self.device),
                torch.zeros(2*self.num_layers, batch_size,self.hidden_dim_word // 2).to(self.device))

    def forward(self, x):

        embeds = x[0]
        pos = x[1]
        masks = x[2]

        # is possible? :
        b,c,l,ed = embeds.shape
        _,_,_,pd = pos.shape
        #embeds = embeds.view(b*c,l,ed)
        #pos = pos.view(b*c,l,pd)

        embeds = embeds.view(b*c*l,ed)
        pos = pos.view(b*c*l,pd)

        # dropout?
        embeds = self.dropout(embeds)

        embeds = self.tanh(self.input_layer_embeds(embeds))
        pos = self.tanh(self.input_layer_pos(pos))

        embeds = embeds.view(b*c,l,-1)
        pos = pos.view(b*c,l,-1)


        masks = masks.view(b*c,l).to(embeds.dtype)
        masks = masks.unsqueeze(-1)

        hidden = self.init_hidden(len(embeds))
        lstm_out, hidden = self.lstm(embeds, hidden)

        hidden_pos = self.init_hidden_pos(len(pos))
        lstm_out_pos, hidden_pos = self.lstm_pos(pos, hidden_pos)

        lstm_out = lstm_out * masks
        lstm_out_pos = lstm_out_pos * masks

        #lstm_out : (batch,max_length, hidden_dim_word)
        #lstm_pos : (batch,max_length, hidden_dim_pos)


        lstm_feats = torch.cat([lstm_out,lstm_out_pos],2)
        #lstm_feats : (batch, max_length, hidden_dim_word+pos)

        lstm_feats = self.dropout(lstm_feats)

        aw = lstm_feats.view(b*c*l,-1)
        aw = self.input_layer_att(aw)
        #aw = self.tanh(aw)

        aw = aw.view(b*c,l,-1)

        hidden_att = self.init_hidden_att(len(aw))
        aw,self.hidden_att = self.lstm_att(aw,hidden_att)
        aw = aw.contiguous()

        aw = aw.view(b*c*l,-1)

        aw = self.hidden2att(aw)
        aw = aw.view(b*c,l,1)

        #aw : (batch, max_legnth, 1)
        aw = F.softmax(aw,1)

        x = lstm_feats * aw
        x = torch.sum(x,1)
        # x : (batch, hidden_dim_word+pos)

        x = x.view(b,c,-1)

        # final output : batch * chunk * utterance_vector

        return x


class BiLSTM_Dec_CR2(nn.Module):

    def __init__(self, device,max_length,num_tag,embedding_dim,hidden_dim,num_layers=1,dropout_rate = 0.0):
        super(BiLSTM_Dec_CR2, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.dropout = nn.Dropout(dropout_rate)
        self.tanh = nn.Tanh()

        self.input_layer = nn.Linear(embedding_dim,hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first = True,
                            num_layers=num_layers, dropout=0,bidirectional=True)

        self.hidden2tag = nn.LSTM(hidden_dim+1, num_tag, batch_first = True,
                            num_layers=1, dropout=0,bidirectional=True)
        #self.hidden2tag_direct = nn.Linear(hidden_dim+1,num_tag)
        #self.hidden2tw  = nn.Linear(hidden_dim,3)
        self.num_tag = num_tag
        self.logit = nn.Linear(num_tag*2, num_tag)

        # dummy for load pretrained. not used
        self.aux_hidden2tag = nn.Linear(hidden_dim,num_tag)

    def init_hidden(self,batch_size):
        return (torch.zeros(2*self.num_layers, batch_size, self.hidden_dim // 2).to(self.device),
                torch.zeros(2*self.num_layers, batch_size,self.hidden_dim // 2).to(self.device))

    def init_hidden_logit(self,batch_size):
        return (torch.zeros(2*self.num_layers, batch_size, self.num_tag).to(self.device),
                torch.zeros(2*self.num_layers, batch_size,self.num_tag).to(self.device))

    def forward(self,x):
        callers = x[1]
        x = x[0]
        x = self.dropout(x)

        b,c,ed = x.shape

        x = x.view(b*c,ed)

        x = self.input_layer(x)
        x = self.tanh(x)

        x = x.view(b,c,-1)

        hidden = self.init_hidden(b)
        x, hidden = self.lstm(x,hidden)

        callers = callers.unsqueeze(-1)
        x = torch.cat([callers,x],2)

        # x : b , c, hidden_dim
        hidden_logit = self.init_hidden_logit(b)
        x, hidden_logit = self.hidden2tag(x,hidden_logit)

        x = x.contiguous()
        x = x.view(b*c,-1)
        x = self.logit(x)
        x = x.view(b,c,-1)

        _, c , _ = x.shape
        outputs = []

        for i in range(c):
            y = x[:,i,:]
            outputs.append(y)

        return outputs

class BiLSTM_META_RAMCR2(nn.Module):
    def __init__(self, device, max_length, num_tag, num_chunk,
                embedding_dim, pos_dim, hidden_dim_list,
                num_layers=1, dropout_rate=0):
        super(BiLSTM_META_RAMCR2,self).__init__()

        self.encoder = BiLSTM_Enc_RAM(device,max_length,num_tag,embedding_dim,pos_dim,
                                    hidden_dim_list[0],hidden_dim_list[1],hidden_dim_list[3],num_layers,dropout_rate = 0.0)
        self.decoder = BiLSTM_Dec_CR2(device,max_length,num_tag,hidden_dim_list[0]+hidden_dim_list[1]+1,
                                    hidden_dim_list[2],num_layers,dropout_rate =dropout_rate)

    def formatter(self,data):
        utterances, poses, qtags,callers,masks, labels = data
        poses = poses.to(utterances.dtype)
        qtags = qtags.to(utterances.dtype)
        callers = callers.to(utterances.dtype)
        return [utterances, poses,masks,qtags,callers] ,labels



    def forward(self,x):

        qtags = x[3]
        callers = x[4]
        # qtags : b * c
        x = self.encoder(x[:3])
        qtags = qtags.unsqueeze(-1)

        x = torch.cat([qtags,x],2)

        # x + qtags
        x = self.decoder([x,callers])


        return x


class ChunkCrossEntropyLoss(nn.Module):

    def __init__(self,num_chunk, weights, ignore_index,label_weight=None,factor=6):
        super(ChunkCrossEntropyLoss,self).__init__()
        self.weights = weights
        self.num_chunk = num_chunk
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,weight=label_weight)
        self.factor = factor

    def forward(self, input, target):
        loss = 0
        for i in range(self.num_chunk):
            loss += self.weights[i] * self.criterion(input[i],target[:,i])

        loss = loss  *self.factor / self.num_chunk
        return loss

class ChunkCrossEntropyLoss_VRM(nn.Module):

    def __init__(self,num_chunk, weights, ignore_index,label_weight=None):
        super(ChunkCrossEntropyLoss_VRM,self).__init__()
        self.weights = weights
        self.num_chunk = num_chunk
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,weight=None)
        self.factor = 6
    def forward(self, input, target):
        loss = 0
        for i in range(self.num_chunk):
            for axis in range(6):
                loss += self.criterion(input[i][...,2*axis:2*axis+2],target[:,i][...,axis])

        loss = loss  *self.factor / self.num_chunk
        return loss

def build_bilstm_ram(device,config,multiplier=1):
    model = BiLSTM_META_RAMCR2(
                device,
                config.sent_len,
                config.num_tag,
                config.chunk_size,
                config.embedding_len,
                config.pos_len,
                [config.hidden_dim_word*multiplier,\
                    config.hidden_dim_pos*multiplier,\
                    config.hidden_dim_dec*multiplier,\
                    config.hidden_dim_att],
                config.num_layers,
                config.dropout_rate
                )
    return model
