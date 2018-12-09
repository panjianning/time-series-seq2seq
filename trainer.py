import torch
import torch.nn as nn
from utils import create_dir_if_not_exists
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, model, criterion, optimizer, scheduler,
                 time_series_list, train_valid_split=0.8,
                 num_epochs=30, early_stopping=5, batch_size=64,
                 checkpoint_path="../checkpoints/seq2seq.h5", plot=True, offet_for_plot=50):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.train_valid_split = train_valid_split
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.time_series_list = time_series_list
        self.plot_ts = []
        self.plot = plot
        self.offset = offet_for_plot

        create_dir_if_not_exists(self.checkpoint_path)

        self._make_loader()

    def _make_loader(self):
        ts_train = []
        ts_valid = []
        for i, s in enumerate(self.time_series_list):
            num_train = int(self.train_valid_split * len(s))
            tr, val = s[:num_train], s[num_train - self.model.input_length:]
            ts_train.append(tr)
            ts_valid.append(val)
            if self.plot:
                x = val[i*self.offset:i*self.offset + self.model.input_length]
                y = val[i*self.offset + i*self.model.input_length:
                        i*self.offset + i*self.model.input_length + self.model.output_length]
                self.plot_ts.append((x, y))

            dataset_train = TimeSeriesDataset(ts_train, input_length=self.model.input_length,
                                              output_length=self.model.output_length)
            dataset_valid = TimeSeriesDataset(ts_valid, input_length=self.model.input_length,
                                              output_length=self.model.output_length)
            self.loader_train = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True)
            self.loader_valid = DataLoader(dataset=dataset_valid, batch_size=self.batch_size, shuffle=False)

    def train(self):
        best_validation_loss = np.inf
        best_epoch = 1
        early_stopping_cnt = 1
        if self.plot:
            plt.ion()
            fig = plt.figure()
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            self.model.teacher_forcing = True
            train_loss = 0.0
            self.scheduler.step()
            for batch_idx, (x, y) in enumerate(tqdm(self.loader_train, ascii=True)):
                x = torch.transpose(x, dim0=0, dim1=1)
                y = torch.transpose(y, dim0=0, dim1=1)
                self.optimizer.zero_grad()
                output = self.model(x, y)
                loss = self.criterion(y, output)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            train_loss /= (len(self.loader_train.dataset) * self.model.output_length)
            self.model.eval()
            valid_loss = 0.0
            self.model.teacher_forcing = False
            for batch_idx, (x, y) in enumerate(self.loader_valid):
                x = torch.transpose(x, dim0=0, dim1=1)
                y = torch.transpose(y, dim0=0, dim1=1)
                output = self.model(x)
                loss = self.criterion(y, output)
                valid_loss += loss.item()
            valid_loss /= (len(self.loader_valid.dataset) * self.model.output_length)
            if valid_loss < best_validation_loss:
                best_validation_loss = valid_loss
                best_epoch = epoch
                early_stopping_cnt = 1
                checkpoint = {
                    'epoch': best_epoch,
                    'train_loss': train_loss,
                    'valid_loss': best_validation_loss,
                    'optimizer': self.optimizer.state_dict(),
                    'model': self.model.state_dict()
                }
                torch.save(checkpoint, self.checkpoint_path)
            else:
                early_stopping_cnt += 1
            if early_stopping_cnt > self.early_stopping:
                break
            print('best epoch %3d, training loss: %.6f, '
                  'validation loss: %.6f' % (best_epoch, train_loss, valid_loss))

            if self.plot:
                if epoch > 1:
                    plt.clf()
                for i, (x, y) in enumerate(self.plot_ts):
                    self._plot_sample_result(self.model, x, y, fig, cur=i + 1)
                fig.canvas.draw()
                fig.canvas.flush_events()

        print('best epoch %3d, best_validation loss: %.6f' % (best_epoch, best_validation_loss))
        self.model.load_state_dict(torch.load(self.checkpoint_path)['model'])

        if self.plot:
            plt.ioff()
            plt.clf()
            for i, (x, y) in enumerate(self.plot_ts):
                self._plot_sample_result(self.model, x, y, fig, cur=i + 1)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.show()

    def _plot_sample_result(self, model, x, y, fig, rows=4, cols=1, cur=1):
        inp = x[:, :, np.newaxis]
        inp = torch.FloatTensor(inp)

        model.teacher_forcing = False
        pred = model(inp)
        pred = np.squeeze(pred.detach().numpy())

        ax = fig.add_subplot(rows, cols, cur)
        ax.plot(np.arange(x.shape[0]), x)
        ax.plot(np.arange(x.shape[0], x.shape[0] + y.shape[0]), y)
        ax.plot(np.arange(x.shape[0], x.shape[0] + y.shape[0]), pred)
