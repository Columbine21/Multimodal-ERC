import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle

from fusion import fusionPlugin

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch * seq_len, n_classes
        target -> batch * seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch * seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss

class gloveEmbedding(nn.Module):
    """w/o finetune glove embedding.
    """
    def __init__(self, glove_embedding_path):
        super(gloveEmbedding, self).__init__()
        embedding_matrix = pickle.load(open(glove_embedding_path, 'rb'))
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        self.embedding.weight.requires_grad = False
    
    def forward(self, token_seq):
        return self.embedding(token_seq)

class CNNFeatureExtractor(nn.Module):
    """TextCNN feature exctractor (Conv1d + MaxPooling + linear)"""
    
    def __init__(self, glove_embedding_path, embedding_dim=300, output_size=100, filters=50, kernel_sizes=[3,4,5], dropout=0.5):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = gloveEmbedding(glove_embedding_path)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=K) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size


    def forward(self, x, umask):
        
        num_utt, batch, num_words = x.size()
        
        x = x.type(LongTensor)  # (num_utt, batch, num_words)
        x = x.view(-1, num_words) # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        emb = self.embedding(x) # (num_utt * batch, num_words) -> (num_utt * batch, num_words, embedding_dim) 
        emb = emb.transpose(-2, -1).contiguous() # (num_utt * batch, num_words, embedding_dim)  -> (num_utt * batch, embedding_dim, num_words) 
        
        convoluted = [F.relu(conv(emb)) for conv in self.convs] # [(num_utt * batch, out_channels, num_words-conv1d_kernel+1)]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted] # [(num_utt * batch, out_channels)]
        concated = torch.cat(pooled, 1)
        features = F.relu(self.fc(self.dropout(concated))) # (num_utt * batch, out_channels * len(kernel_size) ) -> (num_utt * batch, output_size)
        features = features.view(num_utt, batch, -1) # (num_utt * batch, output_size) -> (num_utt, batch, output_size)
        mask = umask.unsqueeze(-1).type(FloatTensor) # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1) # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim) #  (num_utt, batch, 1) -> (num_utt, batch, output_size)
        features = (features * mask) # (num_utt, batch, output_size) -> (num_utt, batch, output_size)

        return features

class CnnModel(nn.Module):
    def __init__(self, args):
        super(CnnModel, self).__init__()


        self.cnn_feat_extractor = CNNFeatureExtractor(args.glove_embedding_path, args.embedding_dim,
         args.cnn_output_size, args.cnn_filters, args.cnn_kernel_sizes, args.cnn_dropout)

        self.late_fusion_module = fusionPlugin(args)
        
        self.classifier = nn.Linear(args.utterance_dim, args.n_classes)


    def forward(self, text_f, seq_len, video, audio, party_mask, mask):
        # print(text_f.shape, seq_len.shape, audio.shape, party_mask.shape, mask.shape)
        text_feature = self.cnn_feat_extractor(text_f, mask)
        video = video.transpose(1, 0).contiguous()
        audio = audio.transpose(1, 0).contiguous()
        utterance_feature = self.late_fusion_module(text_feature, audio, video)

        log_prob = F.log_softmax(self.classifier(utterance_feature), 2)
        return log_prob