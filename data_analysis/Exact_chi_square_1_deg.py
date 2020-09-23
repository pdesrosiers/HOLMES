import numpy as np
from scipy.stats import chi2
import scipy as sp
from loglin_model import *
import time
import itertools
import json
import os
import csv
import matplotlib.pyplot as plt


######### tests préliminaires

def mle_multinomial_from_table(cont_table, N):
    """
    Returns the maximum likelihood estimates for the probabilities of the multinomial distribution that generated the
    contingency table.
    :param cont_table: (np.array) a 2X2 contingency table
    :param N: Number of observations in the contingency table TODO could be computed in this function directly
    :return: p_list (list of floats) MLE for the probabilities of the multinomial distribution that generated the
    contingency table.
    """

    p_list = []
    for element in cont_table.flatten():
        p_list.append(element/N)

    return p_list

def multinomial_cont_table(nb_trials, nb_categories):
    """

    :param nb_trials:
    :param nb_categories:
    :return:
    """
    probabilities = [1 / float(nb_categories)] * nb_categories
    return np.random.multinomial(nb_trials, probabilities, 1).reshape(2, 2)

def multinomial_problist_cont_table(nb_trials, prob_list, s=1):
    return np.random.multinomial(nb_trials, prob_list, s).reshape(s, 2, 2)

def multinomial_problist_cont_cube(nb_trials, prob_list, s=1):
    return np.random.multinomial(nb_trials, prob_list, s).reshape(s, 2, 2, 2)

def multinomial_cont_cube(nb_trials, nb_categories):
    probabilities = [1 / float(nb_categories)] * nb_categories
    return np.random.multinomial(nb_trials, probabilities, 1).reshape(2, 2, 2)


def sample_mult_cont_table(nb_samples, nb_trials, nb_categories):
    samples = []
    for i in range(nb_samples):
        samples.append(multinomial_cont_table(nb_trials, nb_categories))
    return samples

def sample_mult_cont_table_prob(nb_samples, nb_trials, problist):
    samples = []
    for i in range(nb_samples):
        samples.append(multinomial_problist_cont_table(nb_trials, problist))
    return samples

def sample_mult_cont_cube(nb_samples, nb_trials, nb_categories):
    samples = []
    for i in range(nb_samples):
        samples.append(multinomial_cont_cube(nb_trials, nb_categories))
    return samples


def chisq_stats(sample_list, observed):
    chisqlist = []
    for sample in sample_list:
        chisqlist.append(chisq_formula(observed, sample))
    return chisqlist

def chisq_formula(cont_tab, expected):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.
    if float(0) in expected:
        test_stat = 0
    else:
        test_stat = np.sum((cont_tab - expected) ** 2 / expected)

    return test_stat


def chisq_formula_vector(cont_tables, expected):
    # Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    # via MLE or iterative proportional fitting.
    return np.nan_to_num(np.sum(np.sum((cont_tables - expected) ** 2 / expected, axis = 1), axis = 1))

def chisq_formula_vector_for_cubes(cont_tables, expected):
    # Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    # via MLE or iterative proportional fitting.

    return np.nan_to_num(np.sum(np.sum(np.sum((cont_tables - expected) ** 2 / expected, axis = 1), axis = 1), axis=1))


@jit(nopython=True)
def sampled_chisq_test(cont_table, expected_table, sampled_array):
    if float(0) in expected_table:
        test_stat = 0
        pval = 1
    else:
        test_stat = np.sum((cont_table - expected_table) ** 2 / expected_table)
        cdf = np.sum((sampled_array < test_stat) * 1) / len(sampled_array)
        pval = 1 - cdf
    return test_stat, pval

@jit(nopython=True)
def sampled_chisq_test(cont_table, expected_table, sampled_array):
    if float(0) in expected_table:
        test_stat = 0
        pval = 1
    else:
        test_stat = np.sum((cont_table - expected_table) ** 2 / expected_table)
        cdf = np.sum((sampled_array < test_stat) * 1) / len(sampled_array)
        pval = 1 - cdf
    return test_stat, pval

def chisq_test(cont_tab, expected):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.

    df = 1
    test_stat = np.sum((cont_tab-expected)**2/expected)
    p_val = chi2.sf(test_stat, df)

    return test_stat, p_val

if __name__ == '__main__':

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

    def to_occurrence_matrix(matrix, savepath=None):
       """
       Transform a matrix into a binary matrix where entries are 1 if the original entry was different from 0.
       Parameters
       ----------
       matrix (np.array)
       savepath (string) : path and filename under which to save the file
       Returns
       -------
           The binary matrix or None if a savepath is specified.
       """
       if savepath is None:
           return (matrix > 0) * 1
       else:
           np.save(savepath, (matrix > 0) * 1)
    #sam = np.random.multinomial(140, np.array([21, 48, 18, 10, 6, 18, 7, 12])/140, 10000000). reshape(10000000, 2, 4)
    #expec = mle_2x2_ind_vector(sam, 140)
    #chisq = np.nan_to_num(chisq_formula_vector(sam, expec))

    matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
    matrix1 = to_occurrence_matrix(matrix1)

    computed_cont_table = get_cont_table(1, 583, matrix1)

    print(computed_cont_table)
    #exit()


    #computed_cont_table = np.array([[1,  0], [0, 37]])
    N = np.sum(computed_cont_table)
    expected1 = mle_2x2_ind(computed_cont_table)
    print(chisq_test(computed_cont_table, expected1))
    problist = mle_multinomial_from_table(expected1)
    start = time.clock()
    sample = multinomial_problist_cont_table(N, problist, 10000000)
    expec = mle_2x2_ind_vector(sample, N)
    chisq = np.nan_to_num(chisq_formula_vector(sample, expec))
    print(sampled_chisq_test(computed_cont_table, expected1, chisq))
    print('Time for one table: ', time.clock()-start)

    #print(sp.stats.kstest(chisq, 'chi2', args=(3,)))
    #exit()


    ##### Différence entre les pvalues
    df = 1

    ##### différences dans l'allure des courbes
    fig, ax = plt.subplots(1, 1)
    bins2 = np.arange(0, max(chisq) + 1, 0.1)


    #plt.hist(chisqlist_chi1, bins=bins2, weights=np.repeat(1/len(chisqlist_chi1), len(chisqlist_chi1)), alpha=0.5, label='EXACT')
    plt.hist(chisq, bins=bins2, density=True, alpha=0.5, label='exact')
    x = np.arange(0, 100, 0.01)
    # ax.plot(x, chi2.pdf(x, df), 'r-', label='chi2 pdf')
    ax.plot(x, chi2.pdf(x, df), label='Asympt degree 1')
    #ax.plot(x, chi2.pdf(x, 1), label='degree 1')
    # bins2 = np.arange(0, max(chisqlist) + 5, 1)
    # print(max(chisqlist))
    # plt.hist(chisqlist, bins=bins2, alpha=0.5)
    plt.legend()
    plt.xlim([0, 20])
    plt.ylim([0, 1])
    plt.show()

    cdflist = []
    for stat in np.arange(0.1, 20, 0.1):
       cdflist.append(np.sum((chisq < stat)*1)/len(chisq))
       #difflist.append(np.abs(chi2.cdf(stat, 1) - np.sum((chisqlist < stat)*1)/1000000))
       #difflist.append(chi2.cdf(stat, 1) - np.sum((chisqlist < stat) * 1) / 1000000)
       #print('Sampled cumulative prob for chi = ' + str(stat) + 'is :', np.sum((chisqlist < stat)*1)/1000000, 'CHI2 cumulative prob : ', chi2.cdf(stat, 1))
    print('Probability time : ', time.time() - start)
    #plt.plot(np.arange(0.1, 100, 0.1), difflist)
    plt.plot(np.arange(0.1, 20, 0.1), cdflist)
    plt.plot(np.arange(0.1, 20, 0.1), chi2.cdf(np.arange(0.1, 20, 0.1), 1))
    print(cdflist[-1])

    plt.show()

    exit()




    ##################### Find all the pvalues with exact distribution for all
    ##################### tables in the data :






    ########### Count number of different cubes. NEED TO HAVE A CSV FILE
    def get_cont_cube(u_idx, v_idx, w_idx, matrix):
        # Computes the 2X2X2 contingency table for the occurrence matrix

        row_u_present = matrix[u_idx, :]
        row_v_present = matrix[v_idx, :]
        row_w_present = matrix[w_idx, :]

        row_u_not = 1 - row_u_present
        row_v_not = 1 - row_v_present
        row_w_not = 1 - row_w_present

        #All present :
        table000 =np.sum(row_u_present*row_v_present*row_w_present)

        # v absent
        table010 = np.sum(row_u_present*row_v_not*row_w_present)

        # u absent
        table100 = np.sum(row_u_not*row_v_present*row_w_present)

        # u absent, v absent
        table110 = np.sum(row_u_not*row_v_not*row_w_present)

        # w absent
        table001 = np.sum(row_u_present*row_v_present*row_w_not)

        # v absent, w absent
        table011 = np.sum(row_u_present*row_v_not*row_w_not)

        # u absent, w absent
        table101 = np.sum(row_u_not*row_v_present*row_w_not)

        # all absent
        table111 = np.sum(row_u_not*row_v_not*row_w_not)

        return np.array([[[table000, table010], [table100, table110]], [[table001, table011], [table101, table111]]], dtype=np.float64)


    matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))

    matrix1 = to_occurrence_matrix(matrix1, savepath=None)

    table_set = set()

    with open('exact_chi1_triangles_001_final_otu.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            computed_cont_table = get_cont_cube(int(row[0]), int(row[1]), int(row[2]), matrix1)
            table_str = str(int(computed_cont_table[0, 0, 0])) + '_' + str(int(computed_cont_table[0, 0, 1])) + '_' + str(int(computed_cont_table[0, 1, 0])) + '_' + str(int(computed_cont_table[0, 1, 1])) + '_' + str(int(computed_cont_table[1, 0, 0])) + '_' + str(int(computed_cont_table[1, 0, 1])) + '_' + str(int(computed_cont_table[1, 1, 0])) + '_' + str(int(computed_cont_table[1, 1, 1]))
            if table_str not in table_set:
                table_set.add(table_str)
    table_set = list(table_set)
    print('How many different tables : ', len(table_set))
    print(table_set)
    #json.dump(table_set, open("cube_list.json", 'w'))
    exit()

    #with open('table_list.json') as json_file:
    #    table_set = json.load(json_file)
    #    pvaldictio = {}
    #    print(len(table_set))

    #   #### From the different tables : generate the chisqdist :


    ### #TODO iterate over all the tables

    ##    # Max index used in range() :
    #    lastrange = 7400
    #    maxrange = 7500
    for it in range(lastrange, maxrange):

        table_id = table_set[it]
        table = np.random.rand(2,2)
        table_id_list = str.split(table_id, '_')
        #print(table_id, table_id_list)
        table[0, 0] = int(table_id_list[0])
        table[0, 1] = int(table_id_list[1])
        table[1, 0] = int(table_id_list[2])
        table[1, 1] = int(table_id_list[3])

        N = np.sum(table)
        expected_original = mle_2x2_ind(table)
        problist = mle_multinomial_from_table(expected_original)
        chisqlist = []
        start = time.clock()
        for it in range(1000000):
            sample = multinomial_problist_cont_table(N, problist)
            expected = mle_2x2_ind(sample)
            chisqlist.append(chisq_formula(sample, expected))

        pvaldictio[table_id] = sampled_chisq_test(table, expected_original, chisqlist)
        print('Time for one it : ', time.clock()-start)

    json.dump(pvaldictio, open("exact_chisq_1deg\deg1_pvaldictio_" + str(lastrange) + "_" + str(maxrange) + ".json", 'w'))

    #exit()

    ############# Build the entire dictionary
    #import os


    #complete_pval_dictionary = {}

    #directory = 'exact_chisq_1deg_cc'
    #for filename in os.listdir(directory):
    #   if filename.endswith(".json"):

    #       with open(os.path.join(directory, filename)) as json_file:
    #           print(os.path.join(directory, filename))
    #           data = json.load(json_file)
    #           complete_pval_dictionary.update(data)

    #json.dump(complete_pval_dictionary, open("exact_chisq_1deg_cc\complete_pval_dictionary_1deg.json", 'w'))

    #exit()

    ##### Load dictio

    #with open(r'exact_chisq/complete_pval_dictionary.json') as jsonfile:
    #   data = json.load(jsonfile)
    #   print(len(data))

    #exit()

    ######## Différence Chi2 réelle, Chi2 multinomial 1/4, Chi2 multinomiale MLE


    #sam = np.array([[8,  3], [7, 20]])
    #expected1 = mle_2x2_ind(sam)
    #problist = mle_multinomial_from_table(expected1)
    #chisqlist_prob = []
    #test = multinomial_problist_cont_table(38, problist, 10000)
    #expec = mle_2x2_ind_vector(test, 38)
    #chisq = np.nan_to_num(chisq_formula_vector(test, expec))
    #print(sampled_chisq_test(sam, expected1, chisq))
    #start = time.clock()
    #test2 = multinomial_problist_cont_table(38, problist, 10000)
    #print(test[1], test2[1])
    ##test2 = np.random.multinomial(38, problist, 1000).reshape(1000, 2,2)
    #expec2 = mle_2x2_ind_vector(test2, 38)
    #chisq2 = np.nan_to_num(chisq_formula_vector(test2, expec2))
    #print(np.all(chisq == chisq2))
    #print(sampled_chisq_test(sam, expected1, chisq2))

    #for it in range(10000000):
        #sample = multinomial_problist_cont_table(38*1, problist)
        #expected = mle_2x2_ind_vec(sample)
        #if chisq_formula(sample, expected) > 0:
        #    print(sample, expected)
        #chisq_formula(sample, expected)
        #chisqlist_prob.append(chisq_formula(sample, expected))
    #np.save('exact_chi1_table_1_1_6_30', chisqlist_prob)
    #exit()



    ######## Différence Chi2 réelle, Chi2 multinomial 1/4, Chi2 multinomiale MLE
    #samples = sample_mult_cont_table(1000000, 38, 4)
    #pvalist = []
    #for i in range(10):
    #    sam = np.array([[2,  0], [ 0, 36]])*1
    #    expected1 = mle_2x2_ind(sam)
    #    problist = mle_multinomial_from_table(expected1)
    #    chisqlist_prob = []
    #    start = time.clock()
    #    for it in range(1000000):
    #        sample = multinomial_problist_cont_table(38*1, problist)
    #        expected = mle_2x2_ind(sample)
    #        chisqlist_prob.append(chisq_formula(sample, expected))
    #    print(time.clock() - start)
    #    pvalist.append(sampled_chisq_test(sam, expected1, chisqlist_prob))
    #    #np.save('exact_chi1_scaled_1000', chisqlist_prob)

    #print(pvalist)

    #exit()
    #pvalist1000000x10 = [0.0020059999999999523, 0.002047000000000021, 0.0020369999999999555, 0.002055000000000029, 0.002039000000000013, 0.0020839999999999748, 0.0020379999999999843, 0.0020759999999999668, 0.0020569999999999755, 0.0020559999999999468]
    #pvallist = [0.0020999999999999908, 0.0020040000000000058, 0.0019900000000000473, 0.002062666666666657,  0.002019499999999952, 0.0020571999999999813, 0.0020769999999999955,  0.0020481999999999445]
    #plt.plot([100000, 500000, 1000000, 1500000, 2000000, 2500000, 5000000, 10000000], pvallist, 'o')
    #plt.show()

    #TODO : df = 3 semble être plus pr de la distribution asymptotique

    ##### Différence entre les pvalues
    df = 1
    #chisqlist_chi1 = np.load('exact_chi1_scaled_1000.npy')
    chisqlist_chi1 = np.load('exact_chi1_table_1_1_6_30.npy')


    ##### différences dans l'allure des courbes
    fig, ax = plt.subplots(1, 1)
    bins2 = np.arange(0, max(chisqlist_chi1) + 1, 0.1)


    #plt.hist(chisqlist_chi1, bins=bins2, weights=np.repeat(1/len(chisqlist_chi1), len(chisqlist_chi1)), alpha=0.5, label='EXACT')
    plt.hist(chisqlist_chi1, bins=bins2, density=True, alpha=0.5,)
    x = np.arange(0, 100, 0.01)
    # ax.plot(x, chi2.pdf(x, df), 'r-', label='chi2 pdf')
    ax.plot(x, chi2.pdf(x, df), label='Asympt degree 3')
    #ax.plot(x, chi2.pdf(x, 1), label='degree 1')
    # bins2 = np.arange(0, max(chisqlist) + 5, 1)
    # print(max(chisqlist))
    # plt.hist(chisqlist, bins=bins2, alpha=0.5)
    plt.legend()
    plt.xlim([0, 20])
    plt.ylim([0, 1])
    plt.show()




    exit()

