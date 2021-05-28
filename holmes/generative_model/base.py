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

def problist_to_2x2_table(prob_dist, idx1, idx2, sample_size):

    table = np.random.rand(2,2)
    p_00 = 0
    p_10 = 0
    p_01 = 0
    p_11 = 0
    for key in list(prob_dist.keys()):
        if key[idx1] == 0 and key[idx2] == 0:
            p_00 += prob_dist[key]
        elif key[idx1] == 1 and key[idx2] == 0:
            p_10 += prob_dist[key]
        elif key[idx1] == 0 and key[idx2] == 1:
            p_01 += prob_dist[key]
        else:
            p_11 += prob_dist[key]

    table[0, 0] = p_00
    table[1, 0] = p_10
    table[0, 1] = p_01
    table[1, 1] = p_11

    return table * sample_size

def problist_to_2x2x2_cube(prob_dist, idx1, idx2, idx3, sample_size):

    table = np.random.rand(2,2,2)
    p_000 = 0
    p_010 = 0
    p_001 = 0
    p_011 = 0
    p_100 = 0
    p_110 = 0
    p_101 = 0
    p_111 = 0
    for key in list(prob_dist.keys()):
        if key[idx1] == 0 and key[idx2] == 0 and key[idx3] == 0 :
            p_000 += prob_dist[key]
        elif key[idx1] == 1 and key[idx2] == 0 and key[idx3] == 0 :
            p_010 += prob_dist[key]
        elif key[idx1] == 0 and key[idx2] == 1 and key[idx3] == 0 :
            p_001 += prob_dist[key]
        elif key[idx1] == 1 and key[idx2] == 1 and key[idx3] == 0:
            p_011 += prob_dist[key]
        elif key[idx1] == 0 and key[idx2] == 0 and key[idx3] == 1:
            p_100 += prob_dist[key]
        elif key[idx1] == 1 and key[idx2] == 0 and key[idx3] == 1:
            p_110 += prob_dist[key]
        elif key[idx1] == 0 and key[idx2] == 1 and key[idx3] == 1:
            p_101 += prob_dist[key]
        else:
            p_111 += prob_dist[key]

    table[0, 0, 0] = p_000
    table[0, 1, 0] = p_010
    table[0, 0, 1] = p_001
    table[0, 1, 1] = p_011
    table[1, 0, 0] = p_100
    table[1, 1, 0] = p_110
    table[1, 0, 1] = p_101
    table[1, 1, 1] = p_111

    return table * sample_size

class FactorGraph():
    """Original graph that encodes all the dependencies."""

    def __init__(self, facet_list=[], N=400, alpha=0.01, build_sc=False, building_constraint='None'):
        """

        :param facet_list: (list of lists) Each list in the list is a combination of integers
                            representing the nodes of a facet in a simplicial complex.

        :param N: (int) The number of observations on which the algorithms rely to generate random interactions
                  This number should be the same or lower than the number of observations we will generate
                  with this object

        :param alpha: (float) The significance threshold, also used to generate random interactions

        :param build_sc: (bool) If True, the algorithm will make sure that faces of facets are also
                                significance interactions. If False, the algorithm will probably find
                                a structure that is more akin to an hypergraph, where lower order dependencies
                                don't have to be embedded within higher-order dependencies

        :param building_constraint: (str) If 'None', the algorithm will try to respect the facet list, but might destroy
                                          specified dependencies and induce new ones.
                                          If 'Hard', the effective facet list will be identical to the specified
                                          facet_list
                                          If 'Soft', the effective facet list will contain all dependencies specified in
                                          facet_list, but might also contain induced dependencies.
        """
        # build sc stands for Build simplicial complex. It means that, when trying to find probabilities,
        # we only allow probabilities that will make it possible to find empty triangles that will be
        # transformed in 2-simplices when we test the triplet.
        self.build_sc = build_sc
        self.building_constraint = building_constraint
        self.alpha = alpha
        self.N = N
        self.facet_list = facet_list
        self._get_node_list()
        if self.build_sc:
            self.build_simplicial_complex()
        else:
            self.set_factors()
            self.get_dictionary_length_facet_list()
            self.get_effective_facet_list_hypergraph()
            print('Expected : ', self.expected_facet_list_by_length)
            print('Effective : ', self.effective_facet_list)
            for key in range(2, max([max(self.expected_facet_list_by_length), max(self.effective_facet_list)]) + 1):
                print('Induced dependecies of size : ', key)
                try:
                    effective_dep = self.effective_facet_list[key]
                except:
                    effective_dep = {}

                try:
                    expected_dep = self.expected_facet_list_by_length[key]
                except:
                    expected_dep = {}
                print(effective_dep - expected_dep)
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

        facets_for_size = []
        for facet in self.facet_list:
            facet.sort()
            if len(facet) == 1:

                facets_for_size.append(tuple(facet))

        dictionary_length_list[1] = set(facets_for_size)

        self.expected_facet_list_by_length = dictionary_length_list

        return self.expected_facet_list_by_length

    def get_effective_facet_list_hypergraph(self):
        fg_1simplices_list = []
        fg_2simplices_list = []
        simplices_dictio = {}

        probdist = Prob_dist(self)

        largest_facet_size = len(max(self.facet_list, key=len))

        for one_simp in itertools.combinations(self.node_list, 2):

            cont_table = problist_to_2x2_table(probdist.prob_dist, one_simp[0], one_simp[1], self.N)
            expected_1 = mle_2x2_ind(cont_table)
            pval = self.chisq_test_here(cont_table, expected_1)[1]

            if pval < self.alpha:
                fg_1simplices_list.append(one_simp)
        simplices_dictio[2] = set(fg_1simplices_list)

        for two_simp in itertools.combinations(self.node_list, 3):
            cont_cube = problist_to_2x2x2_cube(probdist.prob_dist, two_simp[0], two_simp[1], two_simp[2], self.N)

            expected_2 = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
            if expected_2 is not None:
                pval = self.chisq_test_here(cont_cube, expected_2)[1]
            else:
                pval = 1
            if pval < self.alpha:
                fg_2simplices_list.append(two_simp)

        simplices_dictio[3] = set(fg_2simplices_list)

        self.induced_facet_list = simplices_dictio

        return self.induced_facet_list

    def get_effective_facet_list_cs(self):
        """
        Find the effective simplices of the factor graph using the total probability distribution of states.
        :return: a dictionary where keys are the size of the simplices and the values are set of tuples denoting the
                 simplices in the factorgraph.
        """
        fg_0simplices_list = []
        fg_1simplices_list = []
        fg_2simplices_list = []
        simplices_dictio = {}
        treated_nodes = set()

        self.probdist = Prob_dist(self)

        largest_facet_size = len(max(self.facet_list, key=len))

        for one_simp in itertools.combinations(self.node_list, 2):

            cont_table = problist_to_2x2_table(self.probdist.prob_dist, one_simp[0], one_simp[1], self.N)
            expected_1 = mle_2x2_ind(cont_table)
            pval = self.chisq_test_here(cont_table, expected_1)[1]

            if pval < self.alpha:
                fg_1simplices_list.append(one_simp)
                for node in one_simp:
                    treated_nodes.add(node)
        simplices_dictio[2] = set(fg_1simplices_list)

        for two_simp in itertools.combinations(self.node_list, 3):
            cont_cube = problist_to_2x2x2_cube(self.probdist.prob_dist, two_simp[0], two_simp[1], two_simp[2], self.N)

            expected_2 = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
            if expected_2 is not None:
                pval = self.chisq_test_here(cont_cube, expected_2)[1]
            else:
                pval = 1

            if pval < self.alpha:
                is_a_2simplex = True
                for simplex in itertools.combinations(two_simp, 2):
                    if simplex not in simplices_dictio[2]:
                        is_a_2simplex = False
                        break

                if is_a_2simplex:
                    fg_2simplices_list.append(two_simp)
                    for node in two_simp:
                        treated_nodes.add(node)
        if len(fg_2simplices_list) > 0:
            simplices_dictio[3] = set(fg_2simplices_list)

        independent_nodes = set(self.node_list) - set(treated_nodes)

        for node in independent_nodes:
            fg_0simplices_list.append((node,))

        simplices_dictio[1] = set(fg_0simplices_list)

        self.effective_facet_list = simplices_dictio

        return self.effective_facet_list

    def chisq_test_here(self, cont_tab, expected, df=1):
        # Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
        # via MLE or iterative proportional fitting.
        if np.any(expected == 0):
            return 0, 1
        test_stat = np.sum((cont_tab-expected)**2/expected)
        p_val = chi2.sf(test_stat, df)

        return test_stat, p_val

    def _get_skeleton(self, j=1):

        skeleton_facet_list = []

        for facet in self.facet_list:

            if len(facet) > j + 1:

                for lower_simplex in itertools.combinations(facet, j + 1):

                    skeleton_facet_list.append(list(lower_simplex))

            else:
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
        """
        Iterates through self.facet_list and sets a factor to each facet and coefficients for the factor. The
        coefficients for each factors are determined randomly using set_probabilities_2x2 and set_probabilities_2x2x2.
        TODO : Incomplete. So far it can only manage facets of size 2 to 3 inclusively.
        :return: None. This function sets a list of factor and of coefficient that have to be used in each factors.
        """

        weight_list = []

        factor_list = []

        probability_list = []

        for facet in self.facet_list:

            if len(facet) == 1:

                weight_list.append(-1)

                probability_list.append(self.set_probabilities_2x1())

                factor_list.append(self.onefactor_state)

            elif len(facet) == 2:

                weight_list.append(-1)

                probability_list.append(self.set_probabilities_2x2())

                factor_list.append(self.twofactor_table_entry)


            elif len(facet) == 3:

                weight_list.append(-1)

                probability_list.append(self.set_probabilities_2x2x2())

                factor_list.append(self.threefactor_table_entry)

            else:

                print('Interactions with more than three nodes are not yet coded.')

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
                    if pval < self.alpha and np.count_nonzero(cont_cube) == 8:
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
                pval = self.chisq_test_here(cont_tab, exp)[1]
                if pval < self.alpha and np.count_nonzero(cont_tab) == 4:
                    switch = False

        # print(cont_tab)

        a = np.log(cont_tab[1, 1])
        b = np.log(cont_tab[0, 1])
        c = np.log(cont_tab[1, 0])
        d = np.log(cont_tab[0, 0])

        return [a, b, c, d]

    def set_probabilities_2x1(self):
        """
        Finds appropriate coefficients for a facet of 1 node.
        :return: List of coefficient for a factor related to one random variable
        """

        # print(cont_tab)
        probability_of_presence = np.random.random(1)

        a = np.log(probability_of_presence)
        b = np.log(1-probability_of_presence)

        return [a, b]

    def build_simplicial_complex(self):

        print('Building simplicial complex. If the algorithm has not converged after 100 tries,\n'
              'you\'ll be prompted to continue or stop.')
        self.get_dictionary_length_facet_list()

        switch = True
        i = 1
        while switch:
            if i % 100 == 0:
                go_on = input('The algorithm was not able to find a simplicial complex that respect the constraints.'
                              'Do you wish to go on for 100 more iterations? (type yes or no)')
                if go_on != 'yes':
                    print('Stopping process')
                    break
                else:
                    print('Continuing for 100 more iterations')

            self.set_factors()

            self.get_effective_facet_list_cs()


            if self.building_constraint is not 'None' :
                wrong = False
                for key_size in self.expected_facet_list_by_length:

                    expected = self.expected_facet_list_by_length[key_size]

                    effective = self.effective_facet_list[key_size]
                    #Here we look at the set difference (Expected - effective) and compute its length. If bigger than zero,
                    #expected is not a subset of effective (Effective does not contain all simplices of expected).

                    if len(expected - effective) > 0:
                        wrong = True
                        break
                    else:
                        if self.building_constraint == 'Hard':
                            if len(effective - expected) != 0:
                                wrong = True
                                break

                if not wrong:
                    switch = False
            else:
                switch = False

            i += 1

        print('Done building simplicial complex ' + '(constraint mode : ' + str(self.building_constraint) + ')' )
        print('Expected : ', self.expected_facet_list_by_length)
        print('Effective : ', self.effective_facet_list)
        for key in range(2, max([max(self.expected_facet_list_by_length), max(self.effective_facet_list)]) + 1):
            print('Induced dependecies of size : ', key)
            try:
                effective_dep = self.effective_facet_list[key]
            except:
                effective_dep = {}

            try:
                expected_dep = self.expected_facet_list_by_length[key]
            except:
                expected_dep = {}
            try :
                print(effective_dep - expected_dep)
            except :
                print('None')
            print('Destroyed dependecies of size ', key)
            try:
                print(expected_dep - effective_dep)
            except:
                print('None')

    def set_weight_list(self):

        # TODO

        return

    # For rejection of H0 :[[[62. 19.]  [16. 80.]] [[70. 64.]  [63. 26.]]] [[[77. 12.]  [15. 87.]] [[68. 67.]  [65.  9.]]]
    # Empty triangle to H0 : [[[77.  7.]  [ 9. 91.]] [[63. 70.]  [80.  3.]]]
    def threefactor_table_entry(self, node_states, weight, a=np.log(39), b=np.log(54), c=np.log(85), d=np.log(64), e=np.log(63), f=np.log(19), g=np.log(25), h=np.log(51)):
        """
        Function used to set a factor linking three variables. Parameters are only relevant when sampling the FactorGraph.
        #
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
        :return: (float) value of the factor for a given state and given coefficient
        """
        x1 = node_states[0]
        x2 = node_states[1]

        return weight * (a*x1*x2 + b*(1-x1)*x2 + c*x1*(1-x2) + d*(1-x1)*(1-x2))

    def twofactor_table_entry_pos(self, node_states, weight, a=np.log(0.48), b=0, c=0, d=np.log(1.52)):

        x1 = node_states[0]
        x2 = node_states[1]

        return weight * (a*x1*x2 + b*(1-x1)*x2 + c*x1*(1-x2) + d*(1-x1)*(1-x2))

    def onefactor_state(self, node_states, weight, a=np.log(0.5), b=np.log(0.5)):

        x1 = node_states[0]

        return weight * (a*x1 + b*(1-x1))

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

        if factorgraph is not None :
            self.fg = factorgraph
            self._get_Z()
            self._get_prob_dist()

        else:
            pass


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
        self.probability_distribution = Prob_dist(self.factorgraph)


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
        probability = self.probability_distribution.prob_dist[tuple(new_state)]

        return new_state, probability

    def _propose_bit_flip_metropolis_hasting(self):
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



