from base import *
from metropolis_sampler import *
from loglin_model import *
from another_sign_int import *

def build_bipartite(samples):

    """
    Builds a bipartite matrix from the generated samples. Each column represents a
    random vector, while each row represents a random variable node in the graph
    :param samples: (list of array)
    :return: array in the form of a bipartite matrix
    """

    bipartite = np.random.rand(len(samples[0]), len(samples))

    i = 0

    for sample in samples:
        bipartite[:, i] = sample
        i += 1

    return bipartite

def mle_multinomial_from_table(cont_table):
    n = np.sum(cont_table)
    p_list = []
    for element in cont_table.flatten():
        p_list.append(element/n)

    return p_list

def multinomial_problist_cont_table(nb_trials, prob_list, s=1):
    return np.random.multinomial(nb_trials, prob_list, s).reshape(s, 2, 2)

def chisq_formula_vector(cont_tables, expected):
    # Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    # via MLE or iterative proportional fitting.

    return np.nan_to_num(np.sum(np.sum((cont_tables - expected) ** 2 / expected, axis = 1), axis = 1))

def sampled_chisq_test(cont_table, expected_table, sampled_array):
    if float(0) in expected_table:
        test_stat = 0
        pval = 1
    else:
        test_stat = np.sum((cont_table - expected_table) ** 2 / expected_table)
        cdf = np.sum((sampled_array < test_stat) * 1) / len(sampled_array)
        pval = 1 - cdf
    return test_stat, pval


def phi_coefficient_table(cont_tab):
   row_sums = np.sum(cont_tab, axis=1)
   col_sums = np.sum(cont_tab, axis=0)

   return (cont_tab[0,0]*cont_tab[1,1] - cont_tab[1,0]*cont_tab[0,1])/np.sqrt(row_sums[0]*row_sums[1]*col_sums[0]*col_sums[1])

def chisq_test(cont_tab, expected):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.

    df = 1
    test_stat = np.sum((cont_tab-expected)**2/expected)
    p_val = chi2.sf(test_stat, df)

    return test_stat, p_val

def pairwise_p_values_phi(bipartite_matrix):

    contingency_table = get_cont_table(0, 1, bipartite_matrix)
    expected_table = mle_2x2_ind(contingency_table)
    phi = phi_coefficient_table(contingency_table)
    chi2, p = chisq_test(contingency_table, expected_table)
    return chi2, p, phi

def get_cont_table(u_idx, v_idx, matrix):
    # Computes the 2X2 contingency table for the occurrence matrix
    row_u_present = matrix[u_idx, :]
    row_v_present = matrix[v_idx, :]

    row_u_not_present = 1 - row_u_present
    row_v_not_present = 1 - row_v_present

    # u present, v present
    table00 = np.dot(row_u_present, row_v_present)

    # u present, v NOT present
    table01 = np.dot(row_u_present, row_v_not_present)

    # u NOT present, v present
    table10 = np.dot(row_u_not_present, row_v_present)

    # u NOT present, v NOT present
    table11 = np.dot(row_u_not_present, row_v_not_present)

    return np.array([[table00, table01], [table10, table11]])

if __name__ == '__main__':

    ########## Winning combinaisons :

    """
    factorgraph = FactorGraph([[0, 1, 2]]) ====== factor_list.append(self.threefactor_all_terms_sym_with_diff_weights)

    def threefactor_all_terms_sym_with_diff_weights(self, node_states, weight, a=2, b=1, c=1.2, d=1, e=1, f=1, g=1.7):
    x1 = node_states[0]
    x2 = node_states[1]
    x3 = node_states[2]
    return weight * (a*(x1*x2*x3) + (1-x1)*(1-x2)*(1-x3) + b*x1*x2 + c*x1*x3 + d*x2*x3 + (1-x1)*(1-x2) + (1-x1)*(1-x3) + (1-x2)*(1-x3) + e*x1 + f*x2 + g*x3 + (1-x1) + (1-x2) + (1-x3))
    
    sampler = Sampler(proposer, temperature=1, initial_burn=500, sample_burn=500)
    sampler.sample(2000)    
    
    """
    factorgraph = FactorGraph([[0], [1]])
    print(factorgraph.factor_list)

    #factorgraph = FactorGraph([[0, 1]])

    probdist = Prob_dist(factorgraph)


    print(probdist.prob_dist)


    state = np.random.randint(2, size=2)

    print(state)

    energy_obj = Energy(state, factorgraph)
    proposer = BitFlipProposer(factorgraph, energy_obj, state)
    sampler = Sampler(proposer, temperature=1, initial_burn=100, sample_burn=51)
    sampler.sample(1000)



    #energy = sampler.results['all_energies']
    #print(sampler.results['nb_success'])
    #print(sampler.results['nb_rejected'])
    #print(sampler.results['sample'])
    #plt.plot(energy)
    #plt.show()

    bipartite = build_bipartite(sampler.results['sample'])
    # Rien = 5000
    #1 = 1000
    #2 = 100
    #3 = 50
    #4 = 500
    #5 = 1000
    #7 = 1000
    #8 = 1500
    np.save('bipartite_8', bipartite)
    with open('factorgraph_8.pkl', 'wb') as output:
        pickle.dump(factorgraph, output, pickle.HIGHEST_PROTOCOL)

    exit()






    ###################### OLD ##########################



    #G = nx.Graph()
    #G.add_edge(1, 2, weight=7, color='red')
    #G = nx.read_edgelist("test.edgelist")
    #nx.write_edgelist(G, 'test.edgelist', data=True)
    #nx.write_edgelist(G, 'test.edgelist', data=['color'])
    #nx.write_edgelist(G, 'test.edgelist', data=['color', 'weight'])
    # Use this to generate synthetic data with associated factorgraph object
    #print(G.nodes)
    #with open('factorgraph.pkl', 'rb') as fg_file:
    #    factorgraph = pickle.load(fg_file)

    #print(factorgraph.edges)

    #exit()

    nb_nodes = 2

    G = nx.generators.random_graphs.erdos_renyi_graph(nb_nodes, 1, seed=0)
    factorgraph = FactorGraph(G)

    print('nb edges : ', len(factorgraph.edges))

    nx.draw_networkx(G)
    #plt.show()

    state = np.random.randint(2, size=nb_nodes)
    state = [1,0]
    print(state)
    energy_obj = Energy(state, factorgraph)
    proposer = BitFlipProposer(factorgraph, energy_obj, state)
    print(proposer.total_energy)
    sampler = Sampler(proposer, temperature=1, initial_burn=500, sample_burn=500)
    sampler.sample(1000)

    energy = sampler.results['all_energies']
    #print(sampler.results['nb_success'])
    #print(sampler.results['nb_rejected'])
    print(sampler.results['sample'])
    #plt.plot(energy)
    #plt.show()

    #bipartite = build_bipartite(sampler.results['sample'])


    #np.save('bipartite', bipartite)
    #with open('factorgraph.pkl', 'wb') as output:
    #    pickle.dump(factorgraph, output, pickle.HIGHEST_PROTOCOL)

    #table = get_cont_table(0, 1, bipartite)
    #print(table)

    #N = np.sum(table)
    #expected1 = mle_2x2_ind(table)
    #print(expected1)
    #problist = mle_multinomial_from_table(expected1)
    #start = time.clock()
    #sample = multinomial_problist_cont_table(N, problist, 1000000)
    #expec = mle_2x2_ind_vector(sample, N)
    #chisq = chisq_formula_vector(sample, expec)
    #print(pairwise_p_values_phi(bipartite))
    #print(sampled_chisq_test(table, expected1, chisq))
    #print('Time for one table: ', time.clock() - start)

