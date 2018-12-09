import torch
import numpy as np


class Inference(object):
    def __init__(self, model, time_series_list, checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        self.model = model
        self.time_series_list = time_series_list

    def infer(self):
        predictions = []
        for ts in self.time_series_list[-self.model.input_length:]:
            inp = inp[:,np.newaxis,:]
            inp = torch.FloatTensor(inp)
            self.model.teacher_forcing = False
            prediction = self.model(inp).detach().numpy().reshape(-1,1)
            predictions.append(prediction)
        return predictions

