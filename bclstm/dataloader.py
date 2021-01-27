import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
from tqdm import tqdm

class IEMOCAPDataset(Dataset):

    def __init__(self, dataset_path, mode='train'):

        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(dataset_path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.validVid, self.trainVid = self.trainVid[:12], self.trainVid[12:]
        
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
        return  torch.FloatTensor(self.videoText[vid]), torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])), torch.LongTensor(self.videoLabels[vid]), vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

def IEMOCAPDataLoader(args):
    
    datasets = {
        'train' : IEMOCAPDataset(dataset_path=args.data_path, mode='train'),
        'valid' : IEMOCAPDataset(dataset_path=args.data_path, mode='valid'),
        'test'  : IEMOCAPDataset(dataset_path=args.data_path, mode='test' ) 
    }

    dataLoader = dict()

    dataLoader['train'] = DataLoader(datasets['train'], batch_size=args.batch_size, 
                                            collate_fn=datasets['train'].collate_fn, num_workers=args.num_workers)
    dataLoader['valid'] = DataLoader(datasets['valid'], batch_size=args.batch_size,
                                            collate_fn=datasets['valid'].collate_fn, num_workers=args.num_workers)
    dataLoader['test' ] = DataLoader(datasets['test'], batch_size=args.batch_size,
                                            collate_fn=datasets[' test'].collate_fn, num_workers=args.num_workers)
    
    return dataLoader

class MELDDataset(Dataset):
    def __init__(self, dataset_path, mode='train'):

        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, _ = pickle.load(open(dataset_path, 'rb'))
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.trainVid, self.testVid = list(self.trainVid), list(self.testVid)
        self.validVid, self.trainVid = self.trainVid[1038:], self.trainVid[:1038]

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
        return torch.FloatTensor(self.videoText[vid]), torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])), torch.LongTensor(self.videoLabels[vid]), vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<3 else pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]

def MELDDataLoader(args):
    """
    Returns: For End2End mode: [videoSentence], [utterance_len], [videoAudio], [global_mask], [label], [vid] 
    """
    datasets = {
        'train' : MELDDataset(dataset_path=args.data_path, mode='train'),
        'valid' : MELDDataset(dataset_path=args.data_path, mode='valid'),
        'test'  : MELDDataset(dataset_path=args.data_path, mode='test' ) 
    }

    dataLoader = dict()

    dataLoader['train'] = DataLoader(datasets['train'], batch_size=args.batch_size, 
                                            collate_fn=datasets['train'].collate_fn, num_workers=args.num_workers)
    dataLoader['valid'] = DataLoader(datasets['valid'], batch_size=args.batch_size,
                                            collate_fn=datasets['valid'].collate_fn, num_workers=args.num_workers)
    dataLoader['test' ] = DataLoader(datasets['test'], batch_size=args.batch_size,
                                            collate_fn=datasets[' test'].collate_fn, num_workers=args.num_workers)
    
    return dataLoader