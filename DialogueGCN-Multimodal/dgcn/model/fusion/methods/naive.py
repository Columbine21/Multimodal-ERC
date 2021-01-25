import torch
import torch.nn as nn
import torch.nn.functional as F

class textOnly(nn.Module):

    def __init__(self, args):
        super(textOnly, self).__init__()
        # dimensions are specified in the order of audio, video and text
        # self.text_in, self.video_in, self.audio_in = args.input_features
        # write utterance_dim into args. 
        # args.utterance_dim = self.text_in

    def forward(self, text_x, audio_x, video_x):
        '''
        Args:
            audio_x: tensor of shape (seq_len, batch_size, audio_in)
            video_x: tensor of shape (seq_len, batch_size, video_in)
            text_x: tensor of shape  (seq_len, batch_size, text_in)
        '''
        return text_x


class TriConcat(nn.Module):
    def __init__(self, args):
        super(TriConcat, self).__init__()
        # dimensions are specified in the order of audio, video and text
        # self.text_in, self.video_in, self.audio_in = args.input_features
        # write utterance_dim into args. 
        # args.utterance_dim = self.text_in + self.video_in + self.audio_in

    def forward(self, text_x, audio_x, video_x):
        '''
        Args:
            audio_x: tensor of shape (seq_len, batch_size, audio_in)
            video_x: tensor of shape (seq_len, batch_size, video_in)
            text_x: tensor of shape  (seq_len, batch_size, text_in)
        '''
        return torch.cat([text_x, video_x, audio_x], dim=2)