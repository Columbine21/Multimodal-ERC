import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
from torch.nn.utils.rnn import pad_sequence
from models.Subnets.attentions import SimpleAttention, MatchingAttention
from models.lateFusion.fusionPlugin import fusionPlugin

__all__ = ['DialogueRnn']

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
        # if self.args.cur_time == 2:
        #     print('last_personal_state')
        #     print(last_personal_state)
        batch_size = utterance.size()[0]
        party_count = party_mask.size()[1]
        # speaker_idx : batch, 1
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
        # print('cur_speaker_state')
        # print(cur_speaker_state)
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
        # if self.args.cur_time == 2:
        #     print('cur_personal_state')
        #     print(cur_personal_state)
        last_emotion_state = torch.zeros(batch_size, self.args.emotion_state_dim).type(utterance.type()) \
            if last_emotion_state.size()[0]==0 else last_emotion_state
        # print('last_emotion_state')
        # print(last_emotion_state)
        cur_emotion_state = self.emotion_cell(self._select_parties(cur_personal_state, speaker_idx), last_emotion_state)
        cur_emotion_state = self.dropout(cur_emotion_state)
        # if self.args.cur_time == 2:
        #     print('cur_emotion_state')
        #     print(cur_emotion_state)
        return cur_global, cur_personal_state, cur_emotion_state, attn_weight
class RnnPipeline(nn.Module):
    def __init__(self, args):
        super(RnnPipeline, self).__init__()

        self.args = args
        self.late_fusion_module = fusionPlugin(args)
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
        
        utterance = self.late_fusion_module(text, video, audio)
        
        for utterance_, party_mask_ in zip(utterance, party_mask):
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
            # linear & smax_fc 像是最后的从 p_hidden_state 到最后的 n_classes 的全连接分类器
            self.linear     = nn.Linear(2 * args.emotion_state_dim, 2 * args.hidden_layer_dim)
            self.smax_fc    = nn.Linear(2 * args.hidden_layer_dim, args.n_classes)
            # MatchingAttention module.
            self.matchatt = MatchingAttention(2 * args.emotion_state_dim, 2 * args.emotion_state_dim, att_type='general2')
        else:
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
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2) # seq_len, batch, n_classes
        
        return log_prob