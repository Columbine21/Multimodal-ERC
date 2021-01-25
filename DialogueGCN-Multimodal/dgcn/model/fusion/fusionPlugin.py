import torch
import torch.nn as nn

from dgcn.model.fusion.methods.TFN import TFN
from dgcn.model.fusion.methods.naive import *

__all__ = ['lateFusion']

MODEL_MAP = {
    'tfn': TFN,
    'text': textOnly,
    'concat': TriConcat,
}

class fusionPlugin(nn.Module):
    def __init__(self, args):
        super(fusionPlugin, self).__init__()
        select_model = MODEL_MAP[args.fusion]
        self.Model = select_model(args)

    def forward(self, text, video, audio):
        """Late Fusion Methods takes tri-modalities of each utterances as inputs.

        Args:
            text ([seq_len, batch_size, text_dim]): the contextual representation of text for utterance.
            video ([seq_len, batch_size, video_dim]): the contextual representation of video for utterance.
            audio ([seq_len, batch_size, audio_dim]): the contextual representation of audio for utterance.

        Returns:
            utterance representation ([seq_len, batch_size, utterance_dim]) : the late fusion representation of utterance, 
                                        which contains tri-modal information.
        """

        return self.Model(text, video, audio)