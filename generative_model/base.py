"""
Base class for the implementation of a metropolis scheme for
the generation of of co-occurrence matrices. This code is based
on the architecture of Zoo2019
"""

import numpy as np
import itertools
from scipy.stats import chi2
from loglin_model import iterative_proportional_fitting_AB_AC_BC_no_zeros, mle_2x2_ind
from copy import deepcopy

class FactorGraph():
    """FactoGraph object that encodes all the interactions between the ' random variable ' nodes."""

    def __init__(self, facet_list=[], N=400, alpha=0.01, build_sc=False):
        """

        :param facet_list: facet_list: (list of lists) Each list in the list is a combination of integers
                                    representing the nodes of a facet in a simplicial complex.
        :param N: (int) Number of observations that we plan to generate (used in set_probabilities_2x2 and 2x2x2)
        :param alpha: (float) Threshold of significance
        :param build_sc: (bool) True : forces the factor graph to have a simplicial complex representation (meaning that
                                if the facet [1, 2, 3] exists, faces [1,2] [1,3] and [2,3] also exist.
                                False : The resulting factor graph can be mapped to an hypergraph, but does not guarantee
                                that lower order interactions have to be present for a higher one to exist.
        """
        self.build_sc = build_sc
        self.alpha = alpha
        self.N = N
        self.facet_list = facet_list
        self._get_node_list()
        self.set_factors()
        #print(self.probability_list)


    def chisq_test_fg(self, cont_tab, expected, df=1):
        #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
        #via MLE or iterative proportional fitting.
        if np.any(expected == 0):
            return 0, 1
        test_stat = np.sum((cont_tab-expected)**2/expected)
        p_val = chi2.sf(test_stat, df)

        return test_stat, p_val

    def _get_skeleton(self, j=1):
        """
        Method to obtain all facets of dimension j and bellow. Here we use the word facet, but this function can be used
        with a FactorGraph for which self.build_sc is False. If that's the case though, be carefull when interpreting
        the j-skeleton, since an hypergraph does not possess such structure.
        :param j: (int) dimension of the j-skeleton
        :return: list of all facets of dimension j and bellow. Facets with dimensions higher than j are decomposed into
                 facets of dimension j. E. G. [1,2,3] -> [1,2] [1,3] [2,3] if we select j = 1.
        """

        skeleton_facet_list = []

        for facet in self.facet_list:

            if len(facet) > j + 1 :

                for lower_simplex in itertools.combinations(facet, j + 1):

                    skeleton_facet_list.append(list(lower_simplex))

            else :
                skeleton_facet_list.append(facet)

        return skeleton_facet_list

    def _get_node_list(self):
        """
        :return: The list of 'random variable' (species) nodes in the FactorGraph
        """

        node_set = set()

        for facet in self.facet_list:

            for node in facet:

                node_set.add(node)

        self.node_list = list(node_set)
        self.node_list.sort()


    def set_factors(self):
        """
        Iterates through self.facet_list and sets a factor to each facet and coefficients for the factor. The
        coefficients for each factors are determined randomly using set_probabilities_2x2 and set_probabilities_2x2x2.
        TODO : Incomplete. So far it can only manage facets of size 1 to 3 inclusively.
        TODO : Weight list is a relic of the past and should be deleted
        :return: None. This function sets a list of factor and of coefficient that have to be used in each factors.
        """

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

                print('Interactions with more than three nodes are not yet implemented.')

        self.factor_list = factor_list
        self.weight_list = weight_list
        self.probability_list = probability_list

    def set_probabilities_2x2x2(self):
        """
        Finds appropriate coefficients for a facet of 3 nodes. If self.build_sc is True, we make sure that the
        coefficients will also induce all lower-order interactions. Otherwise, we only find coefficients that make
        it possible to reject the model of no second-order interaction for a give number of observations
        :return: List of coefficient for a factor linking 3 ' random variable ' (species) nodes
        """

        if self.build_sc:
            switch = True
            while switch:
                probs = np.random.rand(8)
                probs = probs/np.sum(probs)
                cont_cube = np.random.multinomial(self.N, probs).reshape((2, 2, 2))
                exp = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
                if exp is not None:
                    pval = self.chisq_test_fg(cont_cube, exp)[1]
                    if pval < self.alpha and np.count_nonzero(cont_cube) == 8:
                        contab1 = np.sum(cont_cube, axis=0)
                        exp1 = mle_2x2_ind(contab1)
                        pval1 = self.chisq_test_fg(contab1, exp1)[1]
                        contab2 = np.sum(cont_cube, axis=1)
                        exp2 = mle_2x2_ind(contab2)
                        pval2 = self.chisq_test_fg(contab2, exp2)[1]
                        contab3 = np.sum(cont_cube, axis=2)
                        exp3 = mle_2x2_ind(contab3)
                        pval3 = self.chisq_test_fg(contab3, exp3)[1]
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
                    pval = self.chisq_test_fg(cont_cube, exp)[1]
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
        """
        Finds appropriate coefficients for a facet of 2 nodes. We find coefficients that make
        it possible to reject the model of independence for a give number of observations
        :return: List of coefficient for a factor linking 2 ' random variable ' (species) nodes
        """

        switch = True
        while switch:
            cont_tab = np.random.multinomial(self.N, [1 / 4] * 4).reshape((2, 2))
            exp = mle_2x2_ind(cont_tab)
            if exp is not None:
                pval = self.chisq_test_fg(cont_tab, exp)[1]
                if pval < self.alpha and np.count_nonzero(cont_tab)==4:
                    switch = False

        a = np.log(cont_tab[1, 1])
        b = np.log(cont_tab[0, 1])
        c = np.log(cont_tab[1, 0])
        d = np.log(cont_tab[0, 0])

        return [a, b, c, d]



    def set_weight_list(self):


        #TODO : Irrelevant and should be removed

        return

    def threefactor_table_entry(self, node_states, weight, a=np.log(39), b=np.log(54), c=np.log(85), d=np.log(64), e=np.log(63), f=np.log(19), g=np.log(25), h=np.log(51)):
        """
        Function used to set a factor linking three variables. Parameters are only relevant when sampling the FactorGraph.

        :param node_states: (int) 0 or 1 if node x_i is present or absent
        :param weight: TODO Irrelevant parameter
        :param a: (float) coefficient for the probability of sampling the state [1 1 1]
        :param b: (float) coefficient for the probability of sampling the state [1 1 0]
        :param c: (float) coefficient for the probability of sampling the state [1 0 1]
        :param d: (float) coefficient for the probability of sampling the state [0 1 1]
        :param e: (float) coefficient for the probability of sampling the state [1 0 0]
        :param f: (float) coefficient for the probability of sampling the state [0 1 0]
        :param g: (float) coefficient for the probability of sampling the state [0 0 1]
        :param h: (float) coefficient for the probability of sampling the state [0 0 0]
        :return: (float) value of the factor for a given state and given coefficient
        """
        x1 = node_states[0]
        x2 = node_states[1]
        x3 = node_states[2]

        return weight * (a*(x1*x2*x3) + b*x1*x2*(1-x3) + c*x1*(1-x2)*x3 + d*(1-x1)*x2*x3 + e*x1*(1-x2)*(1-x3)
                         + f*(1-x1)*x2*(1-x3) + g*(1-x1)*(1-x2)*x3 + h*(1-x1)*(1-x2)*(1-x3))


    def twofactor_table_entry(self, node_states, weight, a=np.log(48), b=np.log(2), c=np.log(2), d=np.log(48)):
        """
        Function used to set a factor linking two variables. Parameters are only relevant when sampling the FactorGraph.
        :param node_states: (int) 0 or 1 if node x_i is present or absent
        :param weight: TODO Irrelevant parameter
        :param a: (float) coefficient for the probability of sampling the state [1 1]
        :param b: (float) coefficient for the probability of sampling the state [0 1]
        :param c: (float) coefficient for the probability of sampling the state [1 0 ]
        :param d: (float) coefficient for the probability of sampling the state [0 0]
        :return:
        """

        x1 = node_states[0]
        x2 = node_states[1]

        return weight * (a*x1*x2 + b*(1-x1)*x2 + c*x1*(1-x2) + d*(1-x1)*(1-x2))


    def onefactor_state(self, node_states, weight):

        return weight * node_states[0]

class Prob_dist():
    """
    Probability distribution object for a given FactorGraph
    """

    def __init__(self, factorgraph, temperature=1):
        """
        Initialize the probability distribution for a given FactorGraph
        :param factorgraph: FactorGraph Object
        :param temperature: TODO irrelevant
        """
        self.temperature = temperature
        self.fg = factorgraph
        self._get_Z()
        self._get_prob_dist()



    def _get_Z(self):
        """
        Computes the partition function of the FactorGraph
        :return: None, it only sets the attribute self.Z and self.energy_per_state
        """

        self.energy_per_state = {}

        self.Z = 0

        for state in itertools.product(range(2), repeat=len(self.fg.node_list)):

            state_energy = Energy(state, self.fg).get_total_energy()

            self.energy_per_state[state] = state_energy

            self.Z += np.exp(-(1/self.temperature)*state_energy)

    def _get_prob_dist(self):
        """
        Computes the probability distribution for the presence/absence states of the FactorGraph.
        :return:
        """

        prob_dist = {}

        for state in itertools.product(range(2), repeat=len(self.fg.node_list)):

            prob_dist[state] = np.exp(-(1/self.temperature)*self.energy_per_state[state])/self.Z

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



