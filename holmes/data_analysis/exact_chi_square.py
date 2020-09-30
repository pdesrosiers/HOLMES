import numpy as np
from scipy.stats import chi2
import scipy as sp
from loglin_model import *
import time
import itertools
import json
import os
import csv


######### tests préliminaires

def mle_multinomial_from_table(cont_table):
    n = np.sum(cont_table)
    p_list = []
    for element in cont_table.flatten():
        p_list.append(element/n)

    return p_list

def multinomial_cont_table(nb_trials, nb_categories):
    probabilities = [1 / float(nb_categories)] * nb_categories
    return np.random.multinomial(nb_trials, probabilities, 1).reshape(2, 2)

def multinomial_problist_cont_table(nb_trials, prob_list):
    return np.random.multinomial(nb_trials, prob_list, 1).reshape(2, 2)

def multinomial_problist_cont_cube(nb_trials, prob_list):
    return np.random.multinomial(nb_trials, prob_list, 1).reshape(2, 2, 2)

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

def sampled_chisq_test(cont_table, expected_table, sampled_array):
    test_stat = np.sum((cont_table - expected_table) ** 2 / expected_table)
    cdf = np.sum((sampled_array < test_stat) * 1) / len(sampled_array)
    pval = 1 - cdf
    return test_stat, pval

def chisq_test(cont_tab, expected):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.

    df = 3
    test_stat = np.sum((cont_tab-expected)**2/expected)
    p_val = chi2.sf(test_stat, df)

    return test_stat, p_val

if __name__ == '__main__':

    ##################### Find all the pvalues with exact distribution for all
    ##################### tables in the data :

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


    #def get_cont_table(u_idx, v_idx, matrix):
    #    # Computes the 2X2 contingency table for the occurrence matrix
    #    row_u_present = matrix[u_idx, :]
    #    row_v_present = matrix[v_idx, :]

    #    row_u_not_present = 1 - row_u_present
    #    row_v_not_present = 1 - row_v_present

    #    # u present, v present
    #    table00 = np.dot(row_u_present, row_v_present)

    #    # u present, v NOT present
    #    table01 = np.dot(row_u_present, row_v_not_present)

    #    # u NOT present, v present
    #    table10 = np.dot(row_u_not_present, row_v_present)

    #    # u NOT present, v NOT present
    #    table11 = np.dot(row_u_not_present, row_v_not_present)

    #    return np.array([[table00, table01], [table10, table11]])
    ###### TO COUNT THE NUMBER OF DIFFERENT TABLES
    matrix1 = np.loadtxt('final_OTU.txt', skiprows=0, usecols=range(1, 39))
    matrix1 = to_occurrence_matrix(matrix1, savepath=None)
    table_set = set()

    ########### Count number of different tables :
    #for one_simplex in itertools.combinations(range(matrix1.shape[0]), 2):
    #    computed_cont_table = get_cont_table(one_simplex[0], one_simplex[1], matrix1)
    #    table_str = str(computed_cont_table[0, 0]) + '_' + str(computed_cont_table[0, 1]) + '_' + str(
    #        computed_cont_table[1, 0]) + '_' + str(computed_cont_table[1, 1])
    #    if table_str not in table_set:
    #        table_set.add(table_str)
    #table_set = list(table_set)
    #print('How many different tables : ', len(table_set))
    #print(table_set)
    #json.dump(table_set, open("table_list.json", 'w'))

    #exit()
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
    #with open('exact_chi1_triangles_001_final_otu.csv', 'r') as csvfile:
    #    reader = csv.reader(csvfile)
    #    next(reader)
    #    for row in reader:
    #        computed_cont_table = get_cont_cube(int(row[0]), int(row[1]), int(row[2]), matrix1)
    #        table_str = str(int(computed_cont_table[0, 0, 0])) + '_' + str(int(computed_cont_table[0, 0, 1])) + '_' + str(int(computed_cont_table[0, 1, 0])) + '_' + str(int(computed_cont_table[0, 1, 1])) + '_' + str(int(computed_cont_table[1, 0, 0])) + '_' + str(int(computed_cont_table[1, 0, 1])) + '_' + str(int(computed_cont_table[1, 1, 0])) + '_' + str(int(computed_cont_table[1, 1, 1]))
    #        if table_str not in table_set:
    #            table_set.add(table_str)
    #table_set = list(table_set)
    #print('How many different tables : ', len(table_set))
    #print(table_set)
    #json.dump(table_set, open("exact_chi1_cube_list.json", 'w'))
    #exit()

    #with open('table_list.json') as json_file:
    #    table_set = json.load(json_file)

    #    #### From the different tables : generate the chisqdist :

    #    pvaldictio = {}
    # #TODO iterate over all the tables

    #    # Max index used in range() :
    #    lastrange = 1950
    #    maxrange = 2000
    #    for it in range(lastrange, maxrange):
    #        table_id = table_set[it]
    #        table = np.random.rand(2,2)
    #        table_id_list = str.split(table_id, '_')
    #        table[0, 0] = int(table_id_list[0])
    #        table[0, 1] = int(table_id_list[1])
    #        table[1, 0] = int(table_id_list[2])
    #        table[1, 1] = int(table_id_list[3])

    #        N = np.sum(table)
    #        expected = mle_2x2_ind(table)
    #        problist = mle_multinomial_from_table(expected)
    #        chisqlist = []
    #        start = time.clock()
    #        for it in range(1000000):
    #            #print(it)
    #            sample = multinomial_problist_cont_table(N, problist)
    #            chisqlist.append(chisq_formula(sample, expected))

    #        pvaldictio[table_id] = sampled_chisq_test(table, expected, chisqlist)
    #        print('Time for one it : ', time.clock()-start)

    #    json.dump(pvaldictio, open("exact_chisq\pvaldictio_" + str(lastrange) + "_" + str(maxrange) + ".json", 'w'))

    def pvalue_AB_AC_BC(cont_cube):
        expected = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
        if expected is not None:
            return chisq_test(cont_cube, expected)[1]
        else:
            return expected


    def chisq_test(cont_tab, expected):
        # Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
        # via MLE or iterative proportional fitting.

        df = 1
        test_stat = np.sum((cont_tab - expected) ** 2 / expected)
        p_val = chi2.sf(test_stat, df)

        return test_stat, p_val

    with open('exact_chi1_cube_list.json') as json_file:
        table_set = json.load(json_file)

        #### From the different tables : generate the chisqdist :

        pvaldictio = {}
        # TODO iterate over all the tables

        # Max index used in range() :
        no_mle_table_count = 0
        for it in range(len(table_set)):
            table_id = table_set[it]
            table = np.random.rand(2, 2, 2)
            table_id_list = str.split(table_id, '_')
            table[0, 0, 0] = int(table_id_list[0])
            table[0, 0, 1] = int(table_id_list[1])
            table[0, 1, 0] = int(table_id_list[2])
            table[0, 1, 1] = int(table_id_list[3])
            table[1, 0, 0] = int(table_id_list[4])
            table[1, 0, 1] = int(table_id_list[5])
            table[1, 1, 0] = int(table_id_list[6])
            table[1, 1, 1] = int(table_id_list[7])
            pval = pvalue_AB_AC_BC(table)
            if pval is not None:
                if pval < 0.001:
                    print(pvalue_AB_AC_BC(table))
                    no_mle_table_count +=1
            #N = np.sum(table)
            #expected_original = iterative_proportional_fitting_AB_AC_BC_no_zeros(table)

            #if expected_original is not None:
            #    print(it)
            #    problist = mle_multinomial_from_table(expected_original)
            #    chisqlist = []
            #    start = time.clock()
            #    for it in range(1000000):
            #        # print(it)
            #        sample = multinomial_problist_cont_cube(N, problist)
            #        expected = iterative_proportional_fitting_AB_AC_BC_no_zeros(sample)
            #        if expected is not None:
            #            chisqlist.append(chisq_formula(sample, expected))
            #        else:
            #            print('Shyte')

            #    pvaldictio[table_id] = sampled_chisq_test(table, expected_original, chisqlist)
            #    print('Time for one it : ', time.clock() - start)
        print(no_mle_table_count)
        exit()

        json.dump(pvaldictio, open("exact_chisq_1deg_cc\cube_pvaldictio.json", 'w'))

            #np.save(os.path.join('exact_chisq', table_id), chisqlist_prob)

    exit()
    ####################################################

    ############## Build the entire dictionary
    #import os
    #with open(r'C:\Users\Xavier\Desktop\Notes de cours\Maîtrise\Projet OTU\Betti_scm-master (1)\Betti_scm-master\exact_chisq\pvaldictio_0_200.json') as jsonfile:
    #    data = json.load(jsonfile)
    #    print(data)

    #complete_pval_dictionary = {}

    #directory = 'exact_chisq'
    #for filename in os.listdir(directory):
    #    if filename.endswith(".json"):

    #        with open(os.path.join(directory, filename)) as json_file:
    #            print(os.path.join(directory, filename))
    #            data = json.load(json_file)
    #            complete_pval_dictionary.update(data)

    #json.dump(complete_pval_dictionary, open("exact_chisq\complete_pval_dictionary.json", 'w'))

    #exit()


    ##### Load dictio

    #with open(r'exact_chisq/complete_pval_dictionary.json') as jsonfile:
    #    data = json.load(jsonfile)
    #    print(len(data))

    #exit()

    ##### Méthode pour extraire la distribution exacte :

    #computed_cont_table = np.array([[2,  0], [ 0, 36]])
    #N = np.sum(computed_cont_table)
    #expected = mle_2x2_ind(computed_cont_table)
    #problist = mle_multinomial_from_table(expected)
    #chisqlist_prob = []

    #for it in range(1000000):
    #   print(it)
    #   sample = multinomial_problist_cont_table(N, problist)
    #   chisqlist_prob.append(chisq_formula(sample, expected))

    #np.save(str(computed_cont_table[0,0]) + '_' + str(computed_cont_table[0,1]) + '_' + str(computed_cont_table[1,0]) + '_' + str(computed_cont_table[1,1]), chisqlist_prob)


    ###### Pour compter le nombre de tables différentes :

    #computed_cont_table = np.array([[2, 0], [0, 36]])
    #table_set = set()

    #table_str = str(computed_cont_table[0, 0]) + '_' + str(computed_cont_table[0, 1]) + '_' + str(
    #    computed_cont_table[1, 0]) + '_' + str(computed_cont_table[1, 1])
    #if table_str not in table_set:
    #    table_set.add(table_str)

    #####

    #exit()


    import matplotlib.pyplot as plt

    #fig, ax = plt.subplots(1, 1)
    #chisqlist = []
    ##for i in range(1000000):
    ##    print(i)
    ##    sample = multinomial_problist_cont_table(38, [1/4, 1/4, 1/4, 1/4])
    ##    expected = mle_2x2_ind(sample)
    ##    chisqlist.append(chisq_formula(sample, expected))

    ##np.save('chisq38', np.array(chisqlist))
    #chisqlist = np.load('chisq38.npy')

    #print(chisqlist)
    #bins6 = np.arange(0, max(chisqlist), 0.1)

    ## print(max(chisqlist))

    #counts, bins = np.histogram(chisqlist, len(np.arange(0, max(chisqlist), 0.5)))
    #print(bins)
    #plt.hist(bins[:-1], bins, weights=counts/np.sum(counts))
    ##plt.show()
    ##hist = plt.hist(chisqlist, bins=bins6, alpha=0.5, label='10mil')

    #x = np.arange(0, 100, 0.01)
    ## ax.plot(x, chi2.pdf(x, df), 'r-', label='chi2 pdf')
    ##ax.plot(x, chi2.pdf(x, df), label='Asympt degree 3')
    #ax.plot(x, chi2.pdf(x, 1), label='degree 1')
    ## bins2 = np.arange(0, max(chisqlist) + 5, 1)
    ## print(max(chisqlist))
    ## plt.hist(chisqlist, bins=bins2, alpha=0.5)
    #plt.legend()
    #plt.xlim([0, 20])
    #plt.ylim([0, 1])
    #plt.show()

    #exit()


    #plist = []
    #for i in range(1000):
    #    contable = multinomial_cont_table(38, 4)
    #    plist.append(mle_multinomial_from_table(contable)[0])
    #print(np.sum(plist)/1000)

    #contable = multinomial_cont_table(38, 4)
    #print(contable)
    #plist = mle_multinomial_from_table(contable)
    #print(plist)
    #exit()



    #fig, ax = plt.subplots(1, 1)
    #x = np.linspace(chi2.ppf(0.01, 1),
    #                chi2.ppf(0.99, 1), 100)
    #ax.plot(x, chi2.pdf(x, 1),
    #        'r-', lw=5, alpha=0.6, label='chi2 pdf')
    #plt.show()

    ######### Temps linéaire :
    #timelist = []
    #itlist = []

    #for it in [1, 10, 100, 1000, 10000, 100000, 1000000, 5000000]:
    #    print(it)
    #    start = time.time()
    #    sample_mult_cont_table(int(it), 38, 4)
    #    timelist.append(time.time()-start)
    #    itlist.append(int(it))
    #plt.plot(itlist, timelist)
    #plt.show()
    #exit()


    ######## Différence Chi2 réelle, Chi2 multinomial 1/4, Chi2 multinomiale MLE
    #samples = sample_mult_cont_table(1000000, 38, 4)


    #sam = np.array([[2,  0], [ 0, 36]])
    #expected = mle_2x2_ind(sam)
    #problist = mle_multinomial_from_table(expected)

    ##chisqlist = []
    #chisqlist_prob = []

    #for it in range(1000000):
    #    print(it)
    #    sample = multinomial_problist_cont_table(38, problist)
    #    chisqlist_prob.append(chisq_formula(sample, expected))
    #for sample in samples:
    #    expected = mle_2x2_ind(sample)
    #    if expected is not None:
    #        chisqlist.append(chisq_formula(sample, expected))

    #for sample in samplesprob:
    #        chisqlist_prob.append(chisq_formula(sample, expected))

    #np.save('exact_chi3_forpval', chisqlist_prob)
    #np.save('exact_chisqlist380000_10millions', chisqlist_prob)
    #exit()

    df = 3
    #TODO : df = 3 semble être plus pr de la distribution asymptotique

    #fig, ax = plt.subplots(1, 1)
    #chisqlist_1000 = np.load('exact_chi3.npy')
    #bins = np.arange(0, max(chisqlist_1000) + 5, 0.1)
    ## print(max(chisqlist))
    #plt.hist(chisqlist_1000, bins=bins, alpha=0.5, density=True, label='un quart')
    #plt.show()


    ##### Différence entre les pvalues
    # np.array([[21, 9], [3, 5]])  # prob
    # np.array([[512, 234], [103, 151]]) #lorsque 1000
    # np.array([[5120, 2340], [1030, 1510]])  # lorsque 10000
    # np.array([[1, 5], [5, 27]]) # probanother
    # np.array([[ 2,  0], [ 0, 36]]) problowpval avec 0 et 753 pval de e-10, phi = 1
    #exact_chisqlist10mil = np.load('exact_chisqlist380000_10millions.npy')
    #exact_chisqlist = np.load('exact_chisqlist.npy')
    #exact_chisqlist380000 = np.load('exact_chisqlist380000.npy')
    #exact_chisqlist38000000 = np.load('exact_chisqlist38000000.npy')
    #chisqlist_unquart = np.load('chisqlist_unquart.npy')
    #chisqlist_prob = np.load('chisqlist_prob.npy')
    #chisqlist_probanother = np.load('chisqlist_probanother.npy')
    chisqlist_chi3 = np.load('exact_chi3.npy')
    chisqlist_chi3_pre = np.load('exact_chi3_precise.npy')
    chisqlist_chi3_forpval = np.load('exact_chi3_forpval.npy')
    cont = np.array([[ 2,  0], [ 0, 36]]) # problowval
    expec = mle_2x2_ind(cont)
    #print(expec)
    #pvalue_exact = sampled_chisq_test(cont, expec, exact_chisqlist)
    #pvalunquart = sampled_chisq_test(cont, expec, chisqlist_unquart)
    #pvalueprob = sampled_chisq_test(cont, expec, chisqlist_prob)
    #pval = chisq_test(cont, expec)
    print(sampled_chisq_test(cont, expec, chisqlist_chi3), sampled_chisq_test(cont, expec, chisqlist_chi3_forpval))
    #print(cont, expec, chisq_test(cont, expec))
    #print(pval, pvalunquart, pvalueprob)
    #print('EXACT : ', pvalue_exact)
    #print('WRONG DIST : ', sampled_chisq_test(cont, expec, chisqlist_problowpval))
    print('ASYMPT : ', chisq_test(cont, expec))
    #exit()

    ##### différences dans l'allure des courbes
    fig, ax = plt.subplots(1, 1)
    #bins = np.arange(0, max(chisqlist_unquart) + 5, 0.1)
    ## print(max(chisqlist))
    #plt.hist(chisqlist_unquart, bins=bins, alpha=0.5, density=True, label='un quart')

    bins2 = np.arange(0, max(chisqlist_chi3) + 1, 0.5)
    bins3 = np.arange(0, max(chisqlist_chi3_pre), 0.05)
    # print(max(chisqlist))
    plt.hist(chisqlist_chi3, bins=bins2, weights=np.repeat(1/len(chisqlist_chi3), len(chisqlist_chi3)), alpha=0.5, label='EXACT')
    plt.hist(chisqlist_chi3_forpval, bins=bins2, weights=np.repeat(1 / len(chisqlist_chi3_forpval), len(chisqlist_chi3_forpval)), alpha=0.5,
             label='EXACT_less')
    #plt.hist(chisqlist_chi3_pre, bins=bins3, alpha=0.5, density=True, label='Precise')

    #bins3 = np.arange(0, max(chisqlist_prob) + 5, 0.1)
    ## print(max(chisqlist))
    #plt.hist(chisqlist_probanother, bins=bins3, alpha=0.5, density=True, label='probanother')

    #bins4 = np.arange(0, max(chisqlist_problowpval) + 5, 0.5)
    # print(max(chisqlist))
    #plt.hist(chisqlist_problowpval, bins=bins4, alpha=0.5, density=True, label='WRONG')

    #bins4 = np.arange(0, max(exact_chisqlist380000) + 5, 0.1)
    # print(max(chisqlist))
    #plt.hist(exact_chisqlist380000, bins=bins4, alpha=0.5, density=True, label='380000')

    #bins5 = np.arange(0, max(exact_chisqlist38000000) + 5, 0.01)
    # print(max(chisqlist))
    #plt.hist(exact_chisqlist38000000, bins=bins5, alpha=0.5, density=True, label='380000')

    #bins6 = np.arange(0, max(exact_chisqlist10mil) + 5, 0.01)
    # print(max(chisqlist))
    #plt.hist(exact_chisqlist10mil, bins=bins6, alpha=0.5, density=True, label='10mil')

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


    #chisqlist = []

    #for sample in samples:
    #    expected = iterative_proportional_fitting_AB_AC_BC_no_zeros(sample, delta=0.000001)
    #    if expected is not None :
    #        chisqlist.append(chisq_formula(sample, expected))

    #print('Sampling time : ', time.time() - start)
    #start = time.time()
    #chisqlist = np.array(chisqlist)
    #np.save('chisqlist2x2x2100000', chisqlist)
    #print('To array time : ', time.time() - start)
    #print(max(chisqlist))

    #exit()
    chisqlist = np.load('chisqlist.npy')
    #stat = 1
    #start = time.time()
    #difflist = []
    #cdflist = []
    #print(len(chisqlist))
    #for stat in np.arange(0.1, 100, 0.1):
    #    cdflist.append(np.sum((chisqlist < stat)*1)/len(chisqlist))
    #    #difflist.append(np.abs(chi2.cdf(stat, 1) - np.sum((chisqlist < stat)*1)/1000000))
    #    #difflist.append(chi2.cdf(stat, 1) - np.sum((chisqlist < stat) * 1) / 1000000)
    #    #print('Sampled cumulative prob for chi = ' + str(stat) + 'is :', np.sum((chisqlist < stat)*1)/1000000, 'CHI2 cumulative prob : ', chi2.cdf(stat, 1))
    #print('Probability time : ', time.time() - start)
    ##plt.plot(np.arange(0.1, 100, 0.1), difflist)
    #plt.plot(np.arange(0.1, 100, 0.1), cdflist)
    #plt.plot(np.arange(0.1, 100, 0.1), chi2.cdf(np.arange(0.1, 100, 0.1), 1))
    #print(cdflist[-1])
    #plt.show()

    print(chisq_formula(np.array([[38,0], [0,0]]),np.array([[38,0], [0,0]])))
    #chisqlist = chisq_stats(samples, np.array([[38,0], [0,0]]))
    #for chi in chisqlist:
    #    print(chi)
    exit()
    #print('HERE')
    #for chi in chisqlist:
    #    if type(chi) != np.float64:
    #        print(chi)

    df = 1
    fig, ax = plt.subplots(1, 1)


    bins = np.arange(0, max(chisqlist)+5, 0.1)
    #print(max(chisqlist))
    plt.hist(chisqlist, bins=bins, alpha=0.5, density=True)

    x = np.arange(min(chisqlist), max(chisqlist), 0.01)
    #ax.plot(x, chi2.pdf(x, df), 'r-', label='chi2 pdf')
    ax.plot(x, chi2.pdf(x, df))
    ax.plot(x, chi2.pdf(x, 1))
    #bins2 = np.arange(0, max(chisqlist) + 5, 1)
    #print(max(chisqlist))
    #plt.hist(chisqlist, bins=bins2, alpha=0.5)
    plt.show()








