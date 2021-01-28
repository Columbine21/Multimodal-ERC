import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
from torch.nn.utils.rnn import pad_sequence

from fusion import fusionPlugin

__all__ = ['DialogueRnn']

# Loss Function
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

# Attention Network 
class SimpleAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M) # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector

        return attn_pool, alpha

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

# Dialogue RNN Section
class DialogueRNNCell(nn.Module):
    def __init__(self, args):
        super(DialogueRNNCell, self).__init__()

        self.args = args

        self.global_cell = nn.GRUCell(args.utterance_dim + args.personal_state_dim, args.global_state_dim)
        self.personal_cell = nn.GRUCell(args.utterance_dim + args.global_state_dim, args.personal_state_dim)
        self.emotion_cell = nn.GRUCell(args.personal_state_dim, args.emotion_state_dim)

        if args.listener_state:
            self.listener_cell = nn.GRUCell(args.utterance_dim + args.personal_state_dim, args.personal_state_dim)
        self.dropout = nn.Dropout(args.dropout)

        if args.context_attention=='simple':
            self.attention = SimpleAttention(args.global_state_dim)
        else:
            self.attention = MatchingAttention(args.global_state_dim, args.utterance_dim, args.concat_attn_dim, args.context_attention)

    def _select_parties(self, X, indices):
        """X: [batch, party, D]; indices: [batch]"""
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel # [batch, D]

    def forward(self, utterance, party_mask, global_hist, last_personal_state, last_emotion_state):
        """
        utterance -> batch, utterance_dim
        party_mask -> batch, party
        global_hist -> t-1, batch, global_state_dim
        last_personal_state -> batch, party, personal_state_dim
        last_emotion_state -> batch, emotion_state_dim
        """

        batch_size = utterance.size()[0]
        party_count = party_mask.size()[1]

        speaker_idx = torch.argmax(party_mask, 1)
        speaker_state = self._select_parties(last_personal_state, speaker_idx) # [batch, personal_state_dim]

        cur_global = self.global_cell(torch.cat([utterance, speaker_state], dim=1),
                torch.zeros(batch_size, self.args.global_state_dim).type(utterance.type()) \
                    if global_hist.size()[0]==0 else global_hist[-1])
        cur_global = self.dropout(cur_global) # [batch, global_state_dim]

        if global_hist.size()[0] == 0:
            speaker_personal_input = torch.zeros(batch_size,self.args.global_state_dim).type(utterance.type())
            attn_weight = None
        else:
            speaker_personal_input, attn_weight = self.attention(global_hist, utterance)

        U_c_ = torch.cat([utterance, speaker_personal_input], dim=1).unsqueeze(1).expand(-1, party_count, -1)
        cur_speaker_state = self.personal_cell(U_c_.contiguous().view(-1, self.args.utterance_dim + self.args.global_state_dim),
                last_personal_state.view(-1, self.args.personal_state_dim)).view(batch_size, -1, self.args.personal_state_dim)
        cur_speaker_state = self.dropout(cur_speaker_state) # [batch, 2, personal_state_dim]

        if self.args.listener_state:
            # U_ : [batch * party, D_m]  (utterance representation.)
            utterance_ = utterance.unsqueeze(1).expand(-1, party_count, -1).contiguous().view(-1, self.args.utterance_dim)
            # ss_ : [batch * party, D_p] (speaker states)
            ss_ = self._select_parties(cur_speaker_state, speaker_idx).unsqueeze(1).\
                    expand(-1, party_count, -1).contiguous().view(-1,self.args.personal_state_dim)
            U_ss_ = torch.cat([utterance_, ss_], 1)
            cur_listener_state = self.listener_cell(U_ss_, last_personal_state.view(-1, self.args.personal_state_dim))\
                .view(batch_size, -1, self.args.personal_state_dim)
            cur_listener_state = self.dropout(cur_listener_state)
        else:
            cur_listener_state = last_personal_state

        party_mask_ = party_mask.unsqueeze(2)
        cur_personal_state = cur_listener_state * (1 - party_mask_) + cur_speaker_state * party_mask_
 
        last_emotion_state = torch.zeros(batch_size, self.args.emotion_state_dim).type(utterance.type()) \
            if last_emotion_state.size()[0]==0 else last_emotion_state

        cur_emotion_state = self.emotion_cell(self._select_parties(cur_personal_state, speaker_idx), last_emotion_state)
        cur_emotion_state = self.dropout(cur_emotion_state)

        return cur_global, cur_personal_state, cur_emotion_state, attn_weight

class RnnPipeline(nn.Module):
    def __init__(self, args):
        super(RnnPipeline, self).__init__()

        self.args = args
        # self.late_fusion_module = fusionPlugin(args)
        self.dropout = nn.Dropout(args.dropout)
        self.dialogue_cell = DialogueRNNCell(args)

    def forward(self, text, video, audio, party_mask):
        # get batch_size, party_num from input.
        batch_size, party_num = party_mask.size()[1], party_mask.size()[2]
        # initialize the global, personal, emotion hidden state.
        global_history_list = torch.zeros(0).type(text.type())
        personal_history = torch.zeros((batch_size, party_num, self.args.personal_state_dim))\
            .type(text.type())
        emotion_history = torch.zeros(0).type(text.type())
        emotion_history_list = emotion_history
        attn_weight_list = []
        
        # utterance = self.late_fusion_module(text, video, audio)
        
        for utterance_, party_mask_ in zip(text, party_mask):
            global_, personal_history, emotion_history, attn_weight_ = self.dialogue_cell(utterance_, party_mask_,\
                 global_history_list, personal_history, emotion_history)
            global_history_list = torch.cat([global_history_list, global_.unsqueeze(0)], dim=0)
            emotion_history_list = torch.cat([emotion_history_list, emotion_history.unsqueeze(0)], dim=0)
            if attn_weight_ is not None:
                attn_weight_list.append(attn_weight_[:,0,:])
        
        return emotion_history_list, attn_weight_list

class DialogueRnn(nn.Module):

    def __init__(self, args):
        super(DialogueRnn, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_rec = nn.Dropout(args.rec_dropout)
        self.attn_on_emotion = args.attn_on_emotion

        # DialogueRNN module.
        self.dialog_rnn_f = RnnPipeline(self.args)
        if args.bi_direction:
            self.dialog_rnn_b = RnnPipeline(self.args)
            # multimodal fusion.
            args.text_fusion_input = 2 * args.emotion_state_dim
            args.audio_fusion_input = args.input_features[1]

            self.late_fusion_module = fusionPlugin(args)
            # linear & smax_fc 像是最后的从 p_hidden_state 到最后的 n_classes 的全连接分类器
            self.linear     = nn.Linear(args.post_fusion_dim, 2 * args.hidden_layer_dim)
            self.smax_fc    = nn.Linear(2 * args.hidden_layer_dim, args.n_classes)
            # MatchingAttention module.
            self.matchatt = MatchingAttention(2 * args.emotion_state_dim, 2 * args.emotion_state_dim, att_type='general2')
        else:
            # multimodal fusion.
            args.text_fusion_input =  args.emotion_state_dim
            args.audio_fusion_input = args.input_features[1]

            self.late_fusion_module = fusionPlugin(args)
            # linear & smax_fc 像是最后的从 p_hidden_state 到最后的 n_classes 的全连接分类器
            self.linear     = nn.Linear(args.emotion_state_dim, args.hidden_layer_dim)
            self.smax_fc    = nn.Linear(args.hidden_layer_dim, args.n_classes)
            # MatchingAttention module.
            self.matchatt = MatchingAttention(args.emotion_state_dim, args.emotion_state_dim, att_type='general2')
    
    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        if X is None:
            # Here for some dataset without certain modalities
            return None
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, text, video, audio, party_mask, mask):
        """
        text -> seq_len, batch, text_dim
        video -> seq_len, batch, video_dim
        audio -> seq_len, batch, audio_dim
        party_mask -> seq_len, batch, party
        """
        emotions_f, alpha_f = self.dialog_rnn_f(text, video, audio, party_mask) # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)

        if self.args.bi_direction:
            rev_text = self._reverse_seq(text, mask)
            rev_video = self._reverse_seq(video, mask)
            rev_audio = self._reverse_seq(audio, mask)
            rev_party_mask = self._reverse_seq(party_mask, mask)
            emotions_b, alpha_b = self.dialog_rnn_b(rev_text, rev_video, rev_audio, rev_party_mask)
            emotions_b = self._reverse_seq(emotions_b, mask)
            emotions_b = self.dropout_rec(emotions_b)
            emotions = torch.cat([emotions_f, emotions_b],dim=-1) # seq_len, batch, 2 * D_e
        else:
            emotions = emotions_f

        if self.attn_on_emotion:
            att_emotions = []
            attn_weight = []
            for emotion_ in emotions:
                att_emotion_, attn_weight_ = self.matchatt(emotions, emotion_, mask=mask)
                att_emotions.append(att_emotion_.unsqueeze(0))
                attn_weight.append(attn_weight_[:,0,:])
            att_emotions = torch.cat(att_emotions, dim=0)

            # add multimodal fusion before the classifier.
            fusion_results = self.late_fusion_module(att_emotions, audio)
            hidden = F.relu(self.linear(fusion_results))
        else:
            fusion_results = self.late_fusion_module(emotions, audio)
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2) # seq_len, batch, n_classes
        
        return log_prob