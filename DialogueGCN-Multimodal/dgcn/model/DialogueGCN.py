import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .EdgeAtt import EdgeAtt
from .GCN import GCN
from .Classifier import Classifier
from .functions import batch_graphify
import dgcn
# TODO import our fusion module.
from .fusion import fusionPlugin

log = dgcn.utils.get_logger()


class DialogueGCN(nn.Module):

    def __init__(self, args):
        super(DialogueGCN, self).__init__()
        # TODO utterance_dim here. 
        if args.dataset == 'iemocap':
            u_dim = 100
            tag_size = 6
        elif args.dataset == 'meld':
            u_dim = 600
            tag_size = 7

        g_dim = 200
        h1_dim = 100
        h2_dim = 100
        hc_dim = 100

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.rnn = SeqContext(u_dim, g_dim, args)
        self.edge_att = EdgeAtt(g_dim, args)
        self.gcn = GCN(g_dim, h1_dim, h2_dim, args)
        
        # TODO import our fusion module.
        args.text_fusion_input, args.audio_fusion_input = g_dim + h2_dim, args.input_features[1]
        self.late_fusion_module = fusionPlugin(args)

        self.clf = Classifier(args.post_fusion_dim, hc_dim, tag_size, args)


        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
        # torch.Size([32]) torch.Size([32, 78, 100]) torch.Size([32, 78, 512]) torch.Size([32, 78, 100])
        # torch.Size([32, 78]) torch.Size([1567])
        # print(data['text_len_tensor'].shape, data['text_tensor'].shape, data['video_tensor'].shape, data['audio_tensor'].shape)
        # print(data['speaker_tensor'].shape, data['label_tensor'].shape)

        # print()
        # print(data['speaker_tensor'])
        # exit()
        
        utterance_rep = data["text_tensor"]

        node_features = self.rnn(data["text_len_tensor"].cpu(), utterance_rep) # [batch_size, mx_len, D_g]
        features, audio, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, data['audio_tensor'], data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)

        graph_out = self.gcn(features, edge_index, edge_norm, edge_type)

        return graph_out, features, audio

    def forward(self, data):
        # print(data["text_tensor"].shape, data["video_tensor"].shape, data["audio_tensor"].shape)
        # exit()
        graph_out, features, audio = self.get_rep(data)
        
        fusion_results = self.late_fusion_module(torch.cat([features, graph_out], dim=-1).unsqueeze(0), audio.unsqueeze(0)).squeeze()
        out = self.clf(fusion_results, data["text_len_tensor"])

        return out

    def get_loss(self, data):
        graph_out, features, audio = self.get_rep(data)
        
        fusion_results = self.late_fusion_module(torch.cat([features, graph_out], dim=-1).unsqueeze(0), audio.unsqueeze(0)).squeeze()
        loss = self.clf.get_loss(fusion_results, data["label_tensor"], data["text_len_tensor"])

        return loss
