import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
from tqdm import tqdm
import re
import json
import unicodedata

class baseTokenizer():
    """Base Tokenizer Class (Inherited by all subclasses) """
    def __init__(self):
        self.PAD_WORD = '[PAD]'
        self.UNK_WORD = '[UNK]'
    
    def unicodeToAscii(self, utterance):
        """ Normalize strings"""
        return ''.join(
            c for c in unicodedata.normalize('NFD', utterance)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, raw_utterance):
        """Remove nonalphabetics for each utterance"""
        str = self.unicodeToAscii(raw_utterance.lower().strip())
        str = re.sub(r"([,.'!?])", r" \1", str)
        str = re.sub(r"[^a-zA-Z,.'!?]+", r" ", str)
        return str
    
    def process(self, utterance):
        pass

class gloveTokenizer(baseTokenizer):
    """Glove Tokenizer for Glove Embedding (End2End Model)"""
    def __init__(self, vocab_path):
        super(gloveTokenizer, self).__init__()
        self.PAD = 0
        self.UNK = 1
        self.word2id = None
        self.loadVocabFromJson(vocab_path)

    def loadVocabFromJson(self, path):
        self.word2id = json.load(open(path))

    def process(self, utterance):
        # baseTokenizer.normalizeString : remove nonalphabetics
        utterance = self.normalizeString(utterance)
        # transform into lower mode.
        wordList = [word.lower() for word in utterance.split()]
        indexes = [self.word2id.get(word, self.UNK) for word in wordList] # unk: 1
        return indexes

class IEMOCAPDataset(Dataset):

    def __init__(self, dataset_path, vocab_path, mode='train'):
        self.tokenizer_ = gloveTokenizer(vocab_path)

        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(dataset_path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.validVid, self.trainVid = self.trainVid[:12], self.trainVid[12:]


        self.utterance_len = dict()
        for dialogue_key in self.videoSentence.keys():
            # word2ids & transform indexes into tensor to use pad_sequence
            self.videoSentence[dialogue_key] = [torch.tensor(self.tokenizer_.process(utterance)).view(-1, 1)
                                                    for utterance in self.videoSentence[dialogue_key]]
            # get each utterance in a dialogue.
            self.utterance_len[dialogue_key] = [len(utterance) for utterance in self.videoSentence[dialogue_key]]
            # padding each utterance in a dialogue into same length. dict: key -> [utterance_num, ]
            self.videoSentence[dialogue_key] = pad_sequence(self.videoSentence[dialogue_key],
                                                            batch_first=True, padding_value=self.tokenizer_.PAD).squeeze()
        
        if mode == 'train':
            self.keys = [x for x in self.trainVid]
        elif mode == 'valid':
            self.keys = [x for x in self.validVid]
        elif mode == 'test':
            self.keys = [x for x in self.testVid ]
        self.keys.sort()
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return  self.videoSentence[vid],\
                torch.FloatTensor(self.utterance_len[vid]),\
                torch.FloatTensor(self.videoVisual[vid]),\
                torch.FloatTensor(self.videoAudio[vid]),\
                torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]]),\
                torch.FloatTensor([1]*len(self.videoLabels[vid])),\
                torch.LongTensor(self.videoLabels[vid]),\
                vid

    def __len__(self):
        return self.len

class IEMOCAPPadCollate:

    def __init__(self, dim=1):
        self.dim = dim

    def pad_tensor(self, vec, pad, dim):

        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size).type(torch.LongTensor)], dim=dim)

    def pad_collate(self, batch):
        """Two stage padding operation. (Used in word level feature.)
                first  stage: padding all utterance in a batch into same max_word_count.
                second stage: padding all dialogue into same max_utter_count.
        """
        # find the length of the longest sequence.
        max_len = max(map(lambda x: x.shape[self.dim], batch))
        
        # pad according to max_len
        batch = [self.pad_tensor(x, pad=max_len, dim=self.dim) for x in batch]
        
        # stack all
        return pad_sequence(batch)
    
    def __call__(self, batch):
        dat = pd.DataFrame(batch)
        
        return [self.pad_collate(dat[i]) if i==0 else \
                pad_sequence(dat[i], True) if i < 7 else \
                dat[i].tolist() for i in dat]

def IEMOCAPDataLoader(args):
    """
    Returns: For End2End mode: [videoSentence], [utterance_len], [videoVisual], [videoAudio], [speaker_mask]
                            [global_mask], [label], [vid] 
             For Features mode : [videoText], [videoVisual], [videoAudio], [speaker_mask], 
                            [global_mask], [label], [vid]
    """
    datasets = {
        'train' : IEMOCAPDataset(dataset_path=args.data_path, vocab_path=args.vocabPath, mode='train'),
        'valid' : IEMOCAPDataset(dataset_path=args.data_path, vocab_path=args.vocabPath, mode='valid'),
        'test'  : IEMOCAPDataset(dataset_path=args.data_path, vocab_path=args.vocabPath, mode='test' ) 
    }

    dataLoader = dict()

    dataLoader['train'] = DataLoader(datasets['train'], batch_size=args.batch_size, 
                                            collate_fn=IEMOCAPPadCollate(dim=1), num_workers=args.num_workers)
    dataLoader['valid'] = DataLoader(datasets['valid'], batch_size=args.batch_size,
                                            collate_fn=IEMOCAPPadCollate(dim=1), num_workers=args.num_workers)
    dataLoader['test' ] = DataLoader(datasets['test'], batch_size=args.batch_size,
                                            collate_fn=IEMOCAPPadCollate(dim=1), num_workers=args.num_workers)
    
    return dataLoader

class MELDDataset(Dataset):
    def __init__(self, dataset_path, vocab_path, mode='train'):
        self.tokenizer_ = gloveTokenizer(vocab_path)

        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, _ = pickle.load(open(dataset_path, 'rb'))
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.trainVid, self.testVid = list(self.trainVid), list(self.testVid)
        self.validVid, self.trainVid = self.trainVid[1038:], self.trainVid[:1038]

        self.utterance_len = dict()
        for dialogue_key in self.videoSentence.keys():
            # word2ids & transform indexes into tensor to use pad_sequence
            self.videoSentence[dialogue_key] = [torch.tensor(self.tokenizer_.process(utterance)).view(-1, 1)
                                                    for utterance in self.videoSentence[dialogue_key]]
            # get each utterance in a dialogue.
            self.utterance_len[dialogue_key] = [len(utterance) for utterance in self.videoSentence[dialogue_key]]
            # padding each utterance in a dialogue into same length. dict: key -> [utterance_num, ]
            self.videoSentence[dialogue_key] = pad_sequence(self.videoSentence[dialogue_key],
                                                            batch_first=True, padding_value=self.tokenizer_.PAD).squeeze()

        if mode == 'train':
            self.keys = [x for x in self.trainVid]
        elif mode == 'valid':
            self.keys = [x for x in self.validVid]
        elif mode == 'test':
            self.keys = [x for x in self.testVid ]
        self.keys.sort()
        self.len = len(self.keys)
    
    def __getitem__(self, index):
        vid = self.keys[index]
        if len(self.utterance_len[vid]) == 1:
            return  self.videoSentence[vid].view(1, -1),\
                    torch.FloatTensor(self.utterance_len[vid]),\
                    torch.FloatTensor(self.videoAudio[vid]),\
                    torch.FloatTensor([1]*len(self.videoLabels[vid])),\
                    torch.LongTensor(self.videoLabels[vid]),\
                    vid
        else:
            return  self.videoSentence[vid],\
                    torch.FloatTensor(self.utterance_len[vid]),\
                    torch.FloatTensor(self.videoAudio[vid]),\
                    torch.FloatTensor([1]*len(self.videoLabels[vid])),\
                    torch.LongTensor(self.videoLabels[vid]),\
                    vid

    def __len__(self):
        return self.len

class MELDPadCollate:

    def __init__(self, dim=1):
        '''instance : [[1,2,3,4,5], [1,4,2]] means a dialogue has 2 utterance and each utterance has 5, 3 words.'''
        self.dim = dim

    def pad_tensor(self, vec, pad, dim):
        '''First stage padding.'''
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size).type(torch.LongTensor)], dim=dim)

    def pad_collate(self, batch):
        """ Two stage padding operation. (Used in word level feature.)
                first  stage: padding all utterance in a batch into same max_word_count.
                second stage: padding all dialogue into same max_utter_count.
        """
        max_len = max(map(lambda x: x.shape[self.dim], batch))
        
        # First stage padding (pad according to max_len)
        batch = [self.pad_tensor(x, pad=max_len, dim=self.dim) for x in batch]
        
        # Second stage padding.
        return pad_sequence(batch)
    
    def __call__(self, batch):
        dat = pd.DataFrame(batch)
        
        return [self.pad_collate(dat[i]) if i==0 else \
                pad_sequence(dat[i], True) if i < 5 else \
                dat[i].tolist() for i in dat]

def MELDDataLoader(args):
    """
    Returns: For End2End mode: [videoSentence], [utterance_len], [videoAudio], [global_mask], [label], [vid] 
    """
    datasets = {
        'train' : MELDDataset(dataset_path=args.data_path, vocab_path=args.vocabPath, mode='train'),
        'valid' : MELDDataset(dataset_path=args.data_path, vocab_path=args.vocabPath, mode='valid'),
        'test'  : MELDDataset(dataset_path=args.data_path, vocab_path=args.vocabPath, mode='test' ) 
    }

    dataLoader = dict()

    dataLoader['train'] = DataLoader(datasets['train'], batch_size=args.batch_size, 
                                            collate_fn=MELDPadCollate(dim=1), num_workers=args.num_workers)
    dataLoader['valid'] = DataLoader(datasets['valid'], batch_size=args.batch_size,
                                            collate_fn=MELDPadCollate(dim=1), num_workers=args.num_workers)
    dataLoader['test' ] = DataLoader(datasets['test'], batch_size=args.batch_size,
                                            collate_fn=MELDPadCollate(dim=1), num_workers=args.num_workers)
    
    return dataLoader

if __name__ == "__main__":
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_path', type=str, default='./dataset/IEMOCAP_features.pkl')
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--vocabPath', type=str, default='./dataset/IEMOCAP_vocab.json')
        return parser.parse_args()

    args = parse_args()
    dataloader = IEMOCAPDataLoader(args)
    with tqdm(dataloader['train']) as td:
        for batch_data in td:
            textf, text_len, visuf, acouf, party_mask, mask, label = batch_data[:-1]

            print(textf.shape)
            print(text_len)
            print(visuf.shape)
            print(acouf.shape)
            print(party_mask.shape)
            print(mask.shape)
            print(label.shape)
            exit()