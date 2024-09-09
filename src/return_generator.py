import torch as t

class ReturnGenerator:
    def __init__(self, base_mean=0.0015, base_volatility=0.18):
        self.base_mean = base_mean
        self.base_volatility = base_volatility

    def generate_returns(self, num_predictions, skill_factor):
        returns = t.normal(self.base_mean * skill_factor, self.base_volatility, (int(num_predictions),))
        return returns.flatten()