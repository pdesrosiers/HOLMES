##############################
# This is a simple script that explains how to infer a simplicial complex or a hypergraph
# from a presence/absence data table, i.e., a binary matrix of dimension (pun), where p
# is the population size (number of random variables) and n is the sample size (number of # data points)

##############################
# Step 0: Prepare the analysis

print("\nStep 0: Load modules and data and set options")

# Load the module for the inference with asymptotic statistical tests
from holmes.data_analysis.asymptotic_significative_interactions import *
import pandas as pd

# Choose the name of the directory (dir_name) where to save the files and the 'prefix' name of each
# created files (data_name)

dir_name = 'results'
data_name = 'web_of_life_TRANSPOSED'

# Create target Directory if doesn't exist

if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    print("Directory ", dir_name, " Created ")
else:
    print("Directory ", dir_name, " already exists")

data_name = os.path.join(dir_name, data_name)

# Choose the significance level alpha to use throughout the analysis.
alpha = 0.01

data_matrix = np.loadtxt(open('M_PL_062.csv', 'r'), delimiter=',')

data_matrix = (data_matrix > 0)*1

data_matrix = data_matrix.T

p,n = data_matrix.shape

print("Preparing to analyze a data set with " + str(p) + " variables and " + str(n) + " samples")

##############################
# First step: Extract all the unique tables

print("\nStep 1: Extract all the unique tables")

# Finds all unique tables
find_unique_tables(data_matrix, data_name)

print("Unique contingency tables saved in file " + data_name + "_table_list.json")

##############################
# Second step: Extract all the pvalues with an asymptotic distribution

print('\nStep 2: Extract pvalues for all tables with an asymptotic distribution')

pvalues_for_tables(data_name)

print("\nResulting p-values saved in file " + data_name + "_asymptotic_pval_dictio.json")

##############################
# Third step: Find table for all links and their associated pvalue

print('\nStep 3 : Find table for all links and their associated pvalue')

with open(data_name + '_asymptotic_pval_dictio.json') as jsonfile:
    dictio = json.load(jsonfile)
    save_pairwise_p_values_phi_dictionary(data_matrix, dictio, data_name + '_asymptotic_pvalues')

print("Results saved in file " + data_name + "_asymptotic_pvalues.csv")

##############################
# Fourth step: Choose alpha and extract the network

print('\nStep 4: Generate network and extract edge_list for a given alpha')

g = read_pairwise_p_values(data_name + '_asymptotic_pvalues.csv', alpha)

nx.write_edgelist(g, data_name + '_asymptotic_edge_list_' + str(alpha)[2:] + '.txt', data=True)

print('Number of nodes : ', g.number_of_nodes())
print('Number of links : ', g.number_of_edges())
print("Edge list saved in file " + data_name + "_asymptotic_edge_list_" + str(alpha)[2:] + ".txt")

##############################
# Fifth step: Find all triangles in the previous network

print('\nStep 5: Find all empty triangles in the network')

g = read_pairwise_p_values(data_name + '_asymptotic_pvalues.csv', alpha)

save_all_triangles(g, data_name + '_asymptotic_triangles_' + str(alpha)[2:])

print('Number of triangles : ', count_triangles_csv(data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '.csv'))

triangles_p_values(data_name, data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '.csv', data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '_pvalues.csv', data_matrix)

##############################
# Sixth step: Extract all 2-simplices

print("\nStep 6: Extract all 2-simplices")

significant_triplet_from_csv(data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '_pvalues.csv', alpha, data_name + '_asymptotic_2-simplices_' + str(alpha)[2:])

print("2-simplices saved in " + data_name + '_asymptotic_2-simplices_' + str(alpha)[2:] + '.csv')

##############################
# Seventh step: Find clicks of size 4 that also form the hull of a tetrahedron

print("\nStep 7: Find clicks of size 4 that also form the hull of a tetrahedron")

g2 = graph_from_2simplices(data_name + '_asymptotic_2-simplices_' + str(alpha)[2:] + '.csv')

save_all_4clics(g2, data_name + '_asymptotic_2-simplices_' + str(alpha)[2:] + '.csv', data_name + '_4clicks')

test_3rd_order_dependencies(data_name + '_4clicks' + '.csv', data_name + '_tetrahedron_pvalue', data_matrix)

extract_3_simplex_from_csv(data_name + '_tetrahedron_pvalue' + '.csv', data_name + '_3-simplices', alpha)

