import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, ts, input_length, output_length):
        self.ts = ts
        self.input_length = input_length
        self.output_length = output_length

        lens = []
        pre = 0
        for s in self.ts:
            cur = self.__len(s) + pre
            lens.append(cur)
            pre = cur
        self.lens = lens

    def __len__(self):
        return self.lens[-1]

    def __getitem__(self, index):
        i = self.__which_ts(index)
        if i > 0:
            index -= self.lens[i-1]
        s = self.ts[i]
        x = s[index:index+self.input_length]
        y = s[index+self.input_length:index+self.input_length + self.output_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)

    def __which_ts(self, index):
        for i, val in enumerate(self.lens):
            if index < val:
                return i

    def __len(self, s):
        return len(s) - self.input_length - self.output_length
