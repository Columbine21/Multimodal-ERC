from tqdm import tqdm
import pandas as pd
import numpy as np, argparse, time, pickle, random, os, datetime

import torch

import torch.optim as optim
from model import MaskedNLLLoss, CnnModel
from dataloader import ERCDataLoader


from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

def setup_seed(seed):
    """ Manually Fix the random seed to get deterministic results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks, losses_sense  = [], [], [], [], []
    
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    # with tqdm(dataloader) as td:
    #     for batch_data in td:
    #         textf, text_len, visuf, acouf, party_mask, mask, label = batch_data[:-1]

            # print(textf.shape)
            # print(text_len)
            # print(visuf.shape)
            # print(acouf.shape)
            # print(party_mask.shape)
            # print(mask.shape)
            # print(label.shape)
            # exit()
    # seed_everything(seed)
    with tqdm(dataloader) as td:
        for data in td:

            if train:
                optimizer.zero_grad()
                
            textf, text_len, visuf, acouf, party_mask, mask, label = [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]
            # print(textf.shape)
            # print(text_len.shape)
            # print(visuf.shape, acouf.shape)
            # print(party_mask.shape, mask.shape, label.shape)
            # exit()
            
            log_prob = model(textf, text_len, visuf, acouf, party_mask, mask)

            lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
            labels_ = label.view(-1) # batch*seq_len
            loss = loss_function(lp_, labels_, mask)

            pred_ = torch.argmax(lp_,1) # batch*seq_len
            preds.append(pred_.data.cpu().numpy())
            labels.append(labels_.data.cpu().numpy())
            masks.append(mask.view(-1).cpu().numpy())
            losses.append(loss.item()*masks[-1].sum())

            if train:
                total_loss = loss
                total_loss.backward()
                
                optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_sense_loss = round(np.sum(losses_sense)/np.sum(masks), 4)

    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, [avg_fscore]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    
    # dataloader settings 
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--vocabPath', type=str, default='./dataset/IEMOCAP_vocab.json')
    parser.add_argument('--data_path', type=str, default='./dataset/IEMOCAP_features.pkl')

    # model settings.
    parser.add_argument('--glove_embedding_path', type=str, default='./dataset/IEMOCAP_embedding.pkl',
                        help='pretrain glove embedding path')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='embedding dims to use')
    parser.add_argument('--cnn_output_size', type=int, default=100)
    parser.add_argument('--cnn_filters', type=int, default=50)
    parser.add_argument('--cnn_kernel_sizes', type=list, default=[3,4,5])
    parser.add_argument('--cnn_dropout', type=float, default=0.5)
    parser.add_argument('--n_classes', type=int, default=6)
    # late fusion module.
    parser.add_argument('--lateFusionModule', type=str, default='tfn')
    parser.add_argument('--input_features', type=tuple, default=(100, 512, 100))
    parser.add_argument('--pre_fusion_hidden_dims', type=tuple, default=(20, 5, 5))
    parser.add_argument('--pre_fusion_dropout', type=float, default=0.4)
    parser.add_argument('--post_fusion_dropout', type=float, default=0.3)

    # train settings.
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0005, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    args.cuda = torch.cuda.is_available()
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    for seed in [1, 11, 111, 1111, 11111]:
        setup_seed(seed)
        args.seed = seed
        
        print(args)

        model = CnnModel(args)
        print('IEMOCAP CNN MODULE ...')

        if args.cuda:
            model.cuda()
        
        loss_weights = torch.FloatTensor([1/0.086747, 1/0.144406, 1/0.227883, 1/0.160585, 1/0.127711, 1/0.252668])
        
        loss_function  = MaskedNLLLoss(loss_weights.cuda() if args.cuda else loss_weights)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        lf = open('logs/cnn_iemocap_logs.txt', 'a')
        
        dataloader = ERCDataLoader(args)

        valid_losses, valid_fscores = [], []
        test_fscores, test_losses = [], []
        best_loss, best_label, best_pred, best_mask = None, None, None, None

        for e in range(args.epochs):
            start_time = time.time()
            train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, dataloader['train'], e, optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, dataloader['valid'], e)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, dataloader['test'], e)
                
            valid_losses.append(valid_loss)
            valid_fscores.append(valid_fscore)
            test_losses.append(test_loss)
            test_fscores.append(test_fscore)
                
            x = 'epoch: {}, train_loss: {}, acc: {}, fscore: {}, valid_loss: {}, acc: {}, fscore: {}, test_loss: {}, acc: {}, fscore: {}, time: {} sec'.format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))
            
            print (x)
            lf.write(x + '\n')

        valid_fscores = np.array(valid_fscores).transpose()
        test_fscores = np.array(test_fscores).transpose()

        score1 = test_fscores[0][np.argmin(valid_losses)]
        score2 = test_fscores[0][np.argmax(valid_fscores[0])]
        scores = [score1, score2]
        scores = [str(item) for item in scores]

        print ('Test Scores: Weighted F1')
        print('@Best Valid Loss: {}'.format(score1))
        print('@Best Valid F1: {}'.format(score2))

        rf = open('results/cosmic_iemocap_results.txt', 'a')
        rf.write('\t'.join(scores) + '\t' + str(args) + '\n')
        rf.close()