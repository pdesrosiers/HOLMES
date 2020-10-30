##############################
# This is a simple script that explains how to infer a simplicial complex or a hypergraph
# from a presence/absence data table, i.e., a binary matrix of dimension (pun), where p 
# is the population size (number of random variables) and n is the sample size (number of # data points)

##############################
# Step 0: Prepare the analysis

print("\nStep 0: Load modules and data and set options")

# Load the module for the inference with asymptotic statistical tests 
from asymptotic_significative_interactions import *

# Set the pption to decide if we use the step method (recommended) or the systematic method, which is
# longer and does not return a simplicial complex. Use step_method = False for the systematic method.

step_method = True 

# Choose the name of the directory (dir_name) where to save the files and the 'prefix' name of each 
# created files (data_name)

dir_name = 'Directory'
data_name = 'Data'

# Create target Directory if doesn't exist

if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    print("Directory ", dir_name, " Created ")
else:
    print("Directory ", dir_name, " already exists")

data_name = os.path.join(dir_name, data_name)

# Choose the significance level alpha to use throughout the analysis.
alpha = 0.01

# Load data
data_matrix = np.load('sample_data.npy')
data_matrix = data_matrix.astype(np.int64)
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
# Fifth step: Extract all the unique cubes

print('\nStep 5: Extract all the unique valid cubes')

find_unique_cubes(data_matrix, data_name)

print("Unique contingency cubes saved in" + data_name + "_cube_list.json")

##############################
# Sixth step: Extract pvalues for all cubes with an asymptotic distribution

print('\nStep 6: Extract pvalues for all cubes with an asymptotic distribution')

pvalues_for_cubes(data_name)

##############################
# Seventh step: Find all triangles in the previous network

print('\nStep 7: Find all empty triangles in the network')

g = read_pairwise_p_values(data_name + '_asymptotic_pvalues.csv', alpha)

save_all_triangles(g, data_name + '_asymptotic_triangles_' + str(alpha)[2:])

print('Number of triangles : ', count_triangles_csv(data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '.csv'))

##############################
# Eighth step: Find all the p-values for the triangles under the hypothesis of homogeneity

print('\nStep 8: Find all the p-values for the triangles under the hypothesis of homogeneity')

with open(data_name + "_asymptotic_cube_pval_dictio.json") as jsonfile:
    dictio = json.load(jsonfile)
    triangles_p_values_tuple_dictionary(data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '.csv', data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '_pvalues.csv', dictio, data_matrix)

##############################
# Last step: Exctract all 2-simplices

print("\nStep 9: Extract all 2-simplices")

significant_triplet_from_csv(data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '_pvalues.csv', alpha, data_name + '_asymptotic_2-simplices_' + str(alpha)[2:])
