import argparse
from sklearn.externals import joblib
from seq2seq import Seq2Seq
from trainer import Trainer
import torch
import torch.nn as nn
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--train_percent', type=float, default=0.8)
    parser.add_argument('--ts_dim', type=int, default=3)
    parser.add_argument('--input_length', type=int, default=24*2*6)
    parser.add_argument('--output_length',type=int, default=24*1*6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size',type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--early_stopping', type=int, default=5)
    parser.add_argument('--checkpoint_path',type=str,default="../checkpoints/seq2seq.h5")

    config = parser.parse_args()

    model = Seq2Seq(hidden_size=config.hidden_size,
                    ts_dim=config.ts_dim,
                    input_length=config.input_length,
                    output_length=config.output_length,
                    teacher_forcing=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=1)

    time_series_list = joblib.load(os.path.join(config.data_path, 'ts.joblib'))

    trainer = Trainer(model=model,
                      time_series_list=time_series_list,
                      criterion=criterion,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      batch_size=config.batch_size,
                      num_epochs=config.num_epochs,
                      early_stopping=config.early_stopping,
                      train_valid_split=config.train_percent,
                      checkpoint_path=config.checkpoint_path,
                      plot=False,
                      offet_for_plot=50)

    trainer.train()


if __name__ == '__main__':
    main()


# python3 run_train.py  --input_length 64 --output_length 16 --data_path ../processed_data/train/