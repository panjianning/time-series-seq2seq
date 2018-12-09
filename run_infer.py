import torch
import numpy as np
from sklearn.externals import joblib
import argparse
from utils import create_dir_if_not_exists
from seq2seq import Seq2Seq
from tqdm import tqdm
import os


class Inference(object):
    def __init__(self, model, time_series_list, checkpoint_path, output_dir):
        model.load_state_dict(torch.load(checkpoint_path)['model'])
        self.model = model
        self.time_series_list = time_series_list
        self.output_dir = output_dir

        create_dir_if_not_exists(self.output_dir)

    def infer(self):
        predictions = []
        for ts in tqdm(self.time_series_list,ascii=True):
            inp = ts[-self.model.input_length:,np.newaxis,:]
            inp = torch.FloatTensor(inp)
            self.model.teacher_forcing = False
            prediction = self.model(inp).detach().numpy()
            predictions.append(prediction)
        joblib.dump(predictions, os.path.join(self.output_dir,'pred_ts.joblib'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="../output/")
    parser.add_argument('--checkpoint_path',type=str,default="../checkpoints/seq2seq.h5")
    parser.add_argument('--ts_path',type=str, default="../processed_data/train/ts.joblib")
    parser.add_argument('--ts_dim', type=int, default=3)
    parser.add_argument('--input_length', type=int, default=64)
    parser.add_argument('--output_length',type=int, default=16)
    parser.add_argument('--hidden_size',type=int, default=128)
    config = parser.parse_args()

    model = Seq2Seq(hidden_size=config.hidden_size,
                    ts_dim=config.ts_dim,
                    input_length=config.input_length,
                    output_length=config.output_length,
                    teacher_forcing=True)

    ts = joblib.load(config.ts_path)
    inferer = Inference(model,ts,output_dir=config.output_dir,checkpoint_path=config.checkpoint_path)
    inferer.infer()


if __name__ == '__main__':
    main()