import argparse

from tqdm import tqdm
import pickle

import dgcn

log = dgcn.utils.get_logger()


def split_iemocap():
    dgcn.utils.set_seed(args.seed)

    video_ids, video_speakers, video_labels, video_text, \
        video_audio, video_visual, video_sentence, trainVids, \
        test_vids = pickle.load(open('data/iemocap/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        train.append(dgcn.Sample(vid, video_speakers[vid], video_labels[vid],
                                 video_text[vid], video_audio[vid], video_visual[vid],
                                 video_sentence[vid]))
    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(dgcn.Sample(vid, video_speakers[vid], video_labels[vid],
                               video_text[vid], video_audio[vid], video_visual[vid],
                               video_sentence[vid]))
    for vid in tqdm(test_vids, desc="test"):
        test.append(dgcn.Sample(vid, video_speakers[vid], video_labels[vid],
                                video_text[vid], video_audio[vid], video_visual[vid],
                                video_sentence[vid]))
    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test

def split_meld():
    dgcn.utils.set_seed(args.seed)

    videoIDs, videoSpeakers, videoLabels, videoText,\
        videoAudio, videoSentence, trainVid, testVid, _ = pickle.load(open('data/meld/MELD_features_raw.pkl', 'rb'))

    train, dev, test = [], [], []
    trainVid, testVid = list(trainVid), list(testVid)
    trainVids, validVids = trainVid[:1038], trainVid[1038:]

    for vid in tqdm(trainVids, desc="train"):
        train.append(dgcn.Sample(vid, videoSpeakers[vid], videoLabels[vid], videoText[vid], videoAudio[vid], None, None))
    for vid in tqdm(validVids, desc="valid"):
        dev.append(dgcn.Sample(vid, videoSpeakers[vid], videoLabels[vid], videoText[vid], videoAudio[vid], None, None))
    for vid in tqdm(testVid, desc="test"):
        test.append(dgcn.Sample(vid, videoSpeakers[vid], videoLabels[vid], videoText[vid], videoAudio[vid], None, None))

    log.info("train vids:")
    log.info(sorted(trainVids))
    log.info("dev vids:")
    log.info(sorted(validVids))
    log.info("test vids:")
    log.info(sorted(testVid))

    return train, dev, test


def main(args):

    if args.dataset == 'iemocap':
        train, dev, test = split_iemocap()
    if args.dataset == 'meld':
        train, dev, test = split_meld()

    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))
    data = {"train": train, "dev": dev, "test": test}
    dgcn.utils.save_pkl(data, args.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["iemocap", "avec", "meld"],
                        help="Dataset name.")
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")
    args = parser.parse_args()

    main(args)
