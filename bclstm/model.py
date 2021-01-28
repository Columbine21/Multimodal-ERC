import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion import fusionPlugin

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

class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            #torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim

        return attn_pool, alpha

class BC_LSTM(nn.Module):
    def __init__(self, args):
        super(BC_LSTM, self).__init__()

        self.args = args

        self.dropout = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(args.utterance_dim, args.emotion_state_dim, num_layers=2, bidirectional=True, dropout=args.dropout)

        if args.attention_type is not None:
            self.matchattn = MatchingAttention(2*args.emotion_state_dim, 2*args.emotion_state_dim, att_type=args.attention_type)
        
        # build fusion arguements.
        args.text_fusion_input = 2 * args.emotion_state_dim
        args.audio_fusion_input = args.input_features[1]

        self.late_fusion_module = fusionPlugin(args)
        self.linear = nn.Linear(args.post_fusion_dim, args.hidden_layer_dim)
        self.smax_fc = nn.Linear(args.hidden_layer_dim, args.n_classes)

    def forward(self, text, video, audio, party_mask, mask):
        """
        modalities -> seq_len, batch, D_x
        qmask -> seq_len, batch, party
        """
        # print(audio.shape)
        contextual_feature, _ = self.lstm(text)

        attn_weights = []

        if self.args.attention_type is not None:
            attn_results = []
            for feature_t in contextual_feature:
                attn_result_t, attn_weight_t = self.matchattn(contextual_feature, feature_t, mask)
                attn_results.append(attn_result_t.unsqueeze(0))
                attn_weights.append(attn_weight_t.squeeze(1))
            attn_results = torch.cat(attn_results, dim=0)
            # attn_weights = torch.cat(attn_weights, dim=0)
            fusion_results = self.late_fusion_module(attn_results, audio)
            hidden = F.relu(self.linear(fusion_results))
        else:
            fusion_results = self.late_fusion_module(contextual_feature, audio)
            hidden = F.relu(self.linear(fusion_results))
        
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)

        return log_prob, attn_weights