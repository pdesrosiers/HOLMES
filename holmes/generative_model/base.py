"""
Base class for the implementation of a metropolis scheme for
the generation of of co-occurrence matrices. This code is based
on the architecture of Zoo2019
"""

import numpy as np
import itertools
from scipy.stats import chi2
from .loglin_model import iterative_proportional_fitting_AB_AC_BC_no_zeros, mle_2x2_ind
from copy import deepcopy

class FactorGraph():
    """Original graph that encodes all the interactions."""

    def __init__(self, facet_list=[], N=400, alpha=0.01, build_sc=False):
        """__init__
        :param facet_list: (list of lists) Each list in the list is a combination of integers
                            representing the nodes of a facet in a simplicial complex.
        """
        # build sc stands for Build simplicial complex. It means that, when trying to find probabilities,
        # we only allow probabilities that will make it possible to find empty triangles that will be
        # transformed in 2-simplices when we test the triplet.
        self.build_sc = build_sc
        self.alpha = alpha
        self.N = N
        self.facet_list = facet_list
        self._get_node_list()
        self.set_factors()
        #print(self.probability_list)

    def get_dictionary_length_facet_list(self):
        """
        Builds a dictionary where keys (int) are lengths and values are sets of simplices of a given size expected in
        the resulting simplicial complex.
        :return:
        """

        dictionary_length_list = {}

        largest_facet_size = len(max(self.facet_list, key=len))


        for size in range(2, largest_facet_size + 1):
            facets_for_size = []
            for facet in self.facet_list:
                facet.sort()
                if len(facet) > size:

                    for lower_simplex in itertools.combinations(facet, size):
                        facets_for_size.append(lower_simplex)

                elif len(facet) == size:
                    facets_for_size.append(tuple(facet))

            dictionary_length_list[size] = set(facets_for_size)

        self.facet_list_by_length = dictionary_length_list

        return self.facet_list_by_length

    def chisq_test_here(self, cont_tab, expected, df=1):
        #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
        #via MLE or iterative proportional fitting.
        if np.any(expected == 0):
            return 0, 1
        test_stat = np.sum((cont_tab-expected)**2/expected)
        p_val = chi2.sf(test_stat, df)

        return test_stat, p_val

    def _get_skeleton(self, j=1):

        skeleton_facet_list = []

        for facet in self.facet_list:

            if len(facet) > j + 1 :

                for lower_simplex in itertools.combinations(facet, j + 1):

                    skeleton_facet_list.append(list(lower_simplex))

            else :
                skeleton_facet_list.append(facet)

        return skeleton_facet_list

    def _get_node_list(self):

        node_set = set()

        for facet in self.facet_list:

            for node in facet:

                node_set.add(node)

        self.node_list = list(node_set)
        self.node_list.sort()


    def set_factors(self):

        weight_list = []

        factor_list = []

        probability_list = []

        for facet in self.facet_list:

            if len(facet) == 1:

                weight_list.append(-1)

                factor_list.append(self.onefactor_state)

            elif len(facet) == 2:

                weight_list.append(-1)

                probability_list.append(self.set_probabilities_2x2())

                factor_list.append(self.twofactor_table_entry)


            elif len(facet) == 3:

                weight_list.append(-1)

                probability_list.append(self.set_probabilities_2x2x2())

                factor_list.append(self.threefactor_table_entry)

            else :

                print('Interactions with more than three nodes are not yet coded.')

        self.factor_list = factor_list
        self.weight_list = weight_list
        self.probability_list = probability_list

    def set_probabilities_2x2x2(self):

        if self.build_sc:
            switch = True
            while switch:
                probs = np.random.rand(8)
                probs = probs/np.sum(probs)
                cont_cube = np.random.multinomial(self.N, probs).reshape((2, 2, 2))
                exp = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
                if exp is not None:
                    pval = self.chisq_test_here(cont_cube, exp)[1]
                    if pval < self.alpha and np.count_nonzero(cont_cube) == 8:
                        contab1 = np.sum(cont_cube, axis=0)
                        exp1 = mle_2x2_ind(contab1)
                        pval1 = self.chisq_test_here(contab1, exp1)[1]
                        contab2 = np.sum(cont_cube, axis=1)
                        exp2 = mle_2x2_ind(contab2)
                        pval2 = self.chisq_test_here(contab2, exp2)[1]
                        contab3 = np.sum(cont_cube, axis=2)
                        exp3 = mle_2x2_ind(contab3)
                        pval3 = self.chisq_test_here(contab3, exp3)[1]
                        if pval1 < self.alpha and pval2 < self.alpha and pval3 < self.alpha:
                            switch = False

        else:
            switch = True
            while switch:
                probs = np.random.rand(8)
                probs = probs / np.sum(probs)
                cont_cube = np.random.multinomial(self.N, probs).reshape((2, 2, 2))
                exp = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
                if exp is not None:
                    pval = self.chisq_test_here(cont_cube, exp)[1]
                    if pval < self.alpha and np.count_nonzero(cont_cube)==8:
                        switch = False


        a = np.log(cont_cube[1, 1, 1])
        b = np.log(cont_cube[1, 1, 0])
        c = np.log(cont_cube[1, 0, 1])
        d = np.log(cont_cube[0, 1, 1])
        e = np.log(cont_cube[1, 0, 0])
        f = np.log(cont_cube[0, 1, 0])
        g = np.log(cont_cube[0, 0, 1])
        h = np.log(cont_cube[0, 0, 0])

        return [a, b, c, d, e, f, g, h]

    def set_probabilities_2x2(self):

        switch = True
        while switch:
            cont_tab = np.random.multinomial(self.N, [1 / 4] * 4).reshape((2, 2))
            exp = mle_2x2_ind(cont_tab)
            if exp is not None:
                pval = self.chisq_test_here(cont_tab, exp)[1]
                if pval < self.alpha and np.count_nonzero(cont_tab)==4:
                    switch = False

        #print(cont_tab)

        a = np.log(cont_tab[1, 1])
        b = np.log(cont_tab[0, 1])
        c = np.log(cont_tab[1, 0])
        d = np.log(cont_tab[0, 0])

        return [a, b, c, d]



    def set_weight_list(self):

        #TODO

        return

    # For rejection of H0 :[[[62. 19.]  [16. 80.]] [[70. 64.]  [63. 26.]]] [[[77. 12.]  [15. 87.]] [[68. 67.]  [65.  9.]]]
    # Empty triangle to H0 : [[[77.  7.]  [ 9. 91.]] [[63. 70.]  [80.  3.]]]
    def threefactor_table_entry(self, node_states, weight, a=np.log(39), b=np.log(54), c=np.log(85), d=np.log(64), e=np.log(63), f=np.log(19), g=np.log(25), h=np.log(51)):
        x1 = node_states[0]
        x2 = node_states[1]
        x3 = node_states[2]

        return weight * (a*(x1*x2*x3) + b*x1*x2*(1-x3) + c*x1*(1-x2)*x3 + d*(1-x1)*x2*x3 + e*x1*(1-x2)*(1-x3)
                         + f*(1-x1)*x2*(1-x3) + g*(1-x1)*(1-x2)*x3 + h*(1-x1)*(1-x2)*(1-x3))


    def twofactor_table_entry(self, node_states, weight, a=np.log(48), b=np.log(2), c=np.log(2), d=np.log(48)):

        x1 = node_states[0]
        x2 = node_states[1]

        return weight * (a*x1*x2 + b*(1-x1)*x2 + c*x1*(1-x2) + d*(1-x1)*(1-x2))

    def twofactor_table_entry_pos(self, node_states, weight, a=np.log(0.48), b=0, c=0, d=np.log(1.52)):

        x1 = node_states[0]
        x2 = node_states[1]

        return weight * (a*x1*x2 + b*(1-x1)*x2 + c*x1*(1-x2) + d*(1-x1)*(1-x2))


    def onefactor_state(self, node_states, weight):

        return weight * node_states[0]

class Prob_dist():

    def __init__(self, factorgraph, temperature=1):

        self.temperature = temperature

        if factorgraph is not None :
            self.fg = factorgraph
            self._get_Z()
            self._get_prob_dist()

        else:
            pass


    def _get_Z(self):

        self.energy_per_state = {}

        self.Z = 0

        for state in itertools.product(range(2), repeat=len(self.fg.node_list)):

            state_energy = Energy(state, self.fg).get_total_energy()

            self.energy_per_state[state] = state_energy

            self.Z += np.exp(-(1/self.temperature)*state_energy)

    def _get_prob_dist(self):

        prob_dist = {}

        for state in itertools.product(range(2), repeat=len(self.fg.node_list)):

            prob_dist[state] = np.exp(-(1/self.temperature)*self.energy_per_state[state])/self.Z

        #prob_dist['T'] = self.temperature

        self.prob_dist = prob_dist



class Energy():
    """Base class that returns the total energy of the state or the local energy of a node at a certain time"""
    def __init__(self, current_state, factorgraph):
        """__init__

        :param current_state: Array of 0 and 1 denoting the current presence and absence state of all nodes
        :param factor graph: FactorGraph object encoding all relations between the nodes
        """
        self.current_state = current_state
        self.factorgraph = factorgraph
        self.total_energy = self.get_total_energy()

    def get_total_energy(self):

        energy = 0

        facet_idx = 0

        for facet in self.factorgraph.facet_list:

            node_states = []

            for node_idx in facet:

                node_states.append(self.current_state[node_idx])

            energy += self.factorgraph.factor_list[facet_idx](node_states, self.factorgraph.weight_list[facet_idx], *self.factorgraph.probability_list[facet_idx])

            facet_idx += 1

        return energy

    def get_local_energy(self, targeted_node):

        energy = 0

        involved_facets_idx = []

        i = 0

        for facet in self.factorgraph.facet_list:

            if targeted_node in facet:

                involved_facets_idx.append(i)

            i += 1


        for facet_idx in involved_facets_idx :

            node_states = []

            for node_idx in self.factorgraph.facet_list[facet_idx]:

                node_states.append(self.current_state[node_idx])

            energy += self.factorgraph.factor_list[facet_idx](node_states, self.factorgraph.weight_list[facet_idx], *self.factorgraph.probability_list[facet_idx])

        return energy

class Proposer():
    """Propose new states for the system and give the energy variation"""
    def __init__(self):
        return

    def __call__(self):
        raise NotImplementedError("__call__ must be overwritten")

    def rejected(self):
        raise NotImplementedError("rejected must be overwritten")

    def get_state(self):
        raise NotImplementedError("get_state must be overwritten")


class BitFlipProposer(Proposer):
    """Propose a new perturbed_graph"""
    def __init__(self, fg, local_transition_energy, state):
        """__init__

        :param g: initial perturbed_graph
        :param local_transition_energy: LocalTransitionEnergy object
        :param prior_energy: PriorEnergy object
        :param p: probability for the geometric series
        """
        super(BitFlipProposer, self).__init__()
        self.state = state
        self.factorgraph = fg
        self.lte = local_transition_energy
        self.total_energy = self.lte.get_total_energy()


    def __call__(self):
        """__call__ propose a new perturbed graph. Returns the energy
        difference

        :returns likelihood_var, bias_var: tuple of energy variation for
        the proposal

        """
        return self._propose_bit_flip()


    def copy(self):
        """copy__ returns a copy of the object

        :returns pgp: PerturbedGraphProposer object with identitical set of
        parameters
        """
        state = self.state
        g = deepcopy(self.g)
        lte = self.lte
        return BitFlipProposer(g, lte,state)


    def _propose_bit_flip(self):
        """_propose_change_point propose a new change point

        :returns likelihood_var, bias_var: tuple of energy variation for
        the proposal
        """

        new_state = np.random.randint(2, size=len(self.state))

        new_total = self.get_total_energy(new_state)

        self.old_state = deepcopy(self.state)

        self.state = new_state

        return new_total

    def get_total_energy(self, state):

        energy = 0

        facet_idx = 0

        for facet in self.factorgraph.facet_list:

            node_states = []

            for node_idx in facet:
                node_states.append(state[node_idx])

            energy += self.factorgraph.factor_list[facet_idx](node_states, self.factorgraph.weight_list[facet_idx], *self.factorgraph.probability_list[facet_idx])

            facet_idx += 1

        return energy

    def get_local_energy(self, targeted_node):


        energy = 0

        involved_facets_idx = []

        i = 0

        for facet in self.factorgraph.facet_list:

            if targeted_node in facet:

                involved_facets_idx.append(i)

            i += 1


        for facet_idx in involved_facets_idx :

            node_states = []

            for node_idx in self.factorgraph.facet_list[facet_idx]:

                node_states.append(self.state[node_idx])

            energy += self.factorgraph.factor_list[facet_idx](node_states, self.factorgraph.weight_list[facet_idx], *self.factorgraph.probability_list[facet_idx])

        return energy

    #def rejected(self):
    #    """rejected reverse applied changes"""

    #    self.state[self.targeted_node] = abs(self.state[self.targeted_node] - 1)

    def rejected(self):
        """rejected reverse applied changes"""

        self.state = self.old_state

    def get_state(self):
        """get_state returns the change point and removed edges set"""
        return deepcopy(self.state)



