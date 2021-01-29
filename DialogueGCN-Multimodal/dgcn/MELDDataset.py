import math
import random
import numpy as np
import torch

class MELDDataset:

    def __init__(self, samples, batch_size):
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        # self.speaker_to_idx = {'M': 0, 'F': 1}

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]

        return batch

    def padding(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s.text) for s in samples]).long()
        mx = torch.max(text_len_tensor).item()

        text_tensor = torch.zeros(batch_size, mx, 600)
        audio_tensor = torch.zeros(batch_size, mx, 300)

        speaker_tensor = torch.zeros((batch_size, mx)).long()

        labels = []
        for i, s in enumerate(samples):
            cur_len = len(s.text)
            
            text_tmp = [torch.from_numpy(t).float() for t in s.text]
            audio_tmp = [torch.from_numpy(t).float() for t in s.audio]

            text_tmp = torch.stack(text_tmp)
            audio_tmp = torch.stack(audio_tmp)

            text_tensor[i, :cur_len, :] = text_tmp
            audio_tensor[i, :cur_len, :] = audio_tmp

            speaker_tensor[i, :cur_len] = torch.tensor([np.argmax(np.array(c)) for c in s.speaker])
            labels.extend(s.label)

        label_tensor = torch.tensor(labels).long()
        data = {
            "text_len_tensor": text_len_tensor,
            "text_tensor": text_tensor,
            "audio_tensor": audio_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor
        }

        return data

    def shuffle(self):
        random.shuffle(self.samples)
