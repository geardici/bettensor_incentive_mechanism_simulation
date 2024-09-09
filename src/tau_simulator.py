import torch as t
import numpy as np

class TauSimulator:
    def __init__(self, num_days, base_tau=1.0, volatility=0.1):
        self.num_days = num_days
        self.base_tau = base_tau
        self.volatility = volatility
        self.tau_values = self._generate_tau_values()

    def _generate_tau_values(self):
        return t.exp(t.cumsum(t.randn(self.num_days) * self.volatility, dim=0)) * self.base_tau

    def get_tau(self, day):
        return self.tau_values[day].item()