from networks.Mlp import Mlp
from data.Data import Data
import os
from training.GradientFlow import GradientFlow
from jax import random
import matplotlib.pyplot as plt
import numpy as np

class ConvergenceExperiment:

    def __init__(self, d, m, h, n, sigma, loss_fn, n_epochs, n_experiments, sigma_w=1.0, sigma_a=1.0, lr=1e-2):
        self.d = d # input dimension
        self.m = m # width
        self.h = h # depth
        self.n = n # number of data points
        self.sigma = sigma
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.n_experiments = n_experiments
        self.sigma_w = sigma_w
        self.sigma_a = sigma_a
        self.lr = lr
    
    def run_single_experiment(self, key_model, key_data):
        model = Mlp(self.d, self.m, self.h, self.sigma, self.sigma_w, self.sigma_a, key_model)
        data = Data(self.d, self.n, key_data)
        gd_flow = GradientFlow(model, data, self.loss_fn, self.lr)
        final_params = gd_flow.train(model.params, self.n_epochs, verbose=False)
        return gd_flow.get_current_loss(final_params)
    
    def run_experiments(self, base_seed=42):
        base_key = random.PRNGKey(base_seed)
        experiment_keys = random.split(base_key, 2*self.n_experiments)
        results = []
        for i in range(self.n_experiments):
            print(f"Running experiment {i+1}/{self.n_experiments}")
            results.append(self.run_single_experiment(experiment_keys[2*i], experiment_keys[2*i + 1]))
        return results
    
    def compute_probability(self, results, threshold=1e-3):
        return np.sum([1 for result in results if result < threshold])/self.n_experiments

if __name__ == '__main__':
    exp = ConvergenceExperiment(3, 500, 200, 10, "relu", "mse", 10000, 1)
    res = exp.run_experiments()
    print(exp.compute_probability(res))