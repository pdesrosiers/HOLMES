from tqdm import tqdm
import numpy as np
from .base import *

class Sampler():
    """Metropolis sampler to identify change point and perturbation"""
    def __init__(self, proposer, temperature=1, initial_burn=1000,
                 sample_burn=100):
        """__init__
        :param proposer: Proposer object
        :param temperature: temperature of the system
        """
        self.proposer = proposer
        self.temperature = temperature
        self.initial_burn = initial_burn
        self.sample_burn = sample_burn
        self.results = dict()
        self.results['sample'] = []
        self.results['all_energies'] = []
        self.results['sampled_energies'] = []
        self.results['nb_success'] = 0
        self.results['nb_rejected'] = 0
        self.thermalized = False

    def _step(self):
        new_energy = self.proposer()

        if new_energy < self.proposer.total_energy:
            p_a = 1
        else:
            p_a = np.exp(-(1/self.temperature)*(new_energy - self.proposer.total_energy))

        u = np.random.random()
        if u > p_a:
            self.proposer.rejected()
            self.results['nb_rejected'] += 1
        else:
            self.proposer.total_energy = new_energy
            self.results['nb_success'] += 1

    def sample(self, size=100):
        nb_samples = 0
        #maxprob = self.proposer.probability_distribution.prob_dist[max(self.proposer.probability_distribution.prob_dist)]
        maxprob = max(self.proposer.probability_distribution.prob_dist.values())
        #print('Maxprob : ', maxprob)

        with tqdm(total=size, desc="Sampling") as pbar:
            while nb_samples < size:

                new_state, probability = self.proposer()

                u = np.random.random()*maxprob
                #u = np.random.uniform(0, maxprob)
                #print(probability, u)
                if u > probability:
                    self.results['nb_rejected'] += 1
                else:
                    self.results['nb_success'] += 1
                    self.results['sample'].append(new_state)

                    nb_samples += 1
                    pbar.update(1)

    def sample_metropolis_hasting(self, size=100):

        if not self.thermalized:
            for iterator in tqdm(range(self.initial_burn),
                                 desc="Thermalization"):
                self._step()
                self.results['all_energies'].append(self.proposer.total_energy)

        with tqdm(total=size, desc="Sampling") as pbar:
            for iterator in range(size*self.sample_burn + 1):
                self._step()

                self.results['all_energies'].append(self.proposer.total_energy)
                if iterator % self.sample_burn == 0 and iterator != 0:
                    self.results['sample'].append(self.proposer.get_state())
                    self.results['sampled_energies'].append(
                        self.proposer.total_energy)
                    pbar.update(1)

    def get_mmse(self):
        """
        :return: minimum mean squared error of the state (self.s, self.removed_edges).
        """
        mmse_s = 0
        mmse_edges = dict()
        n = len(self.results['sample'])
        for state in self.results['sample']:
            mmse_s += state[0]/n
            for edge in state[1]:
                if edge in mmse_edges:
                    # add an occurence of this removed edge to the dictionary
                    mmse_edges[edge] += 1/n
                else:
                    # initialize key (edge) in dictionary if not already there
                    mmse_edges[edge] = 1/n
        return mmse_s, mmse_edges
