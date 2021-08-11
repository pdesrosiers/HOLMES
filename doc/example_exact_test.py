##############################
# This is a simple script that explains how to infer a simplicial complex or a hypergraph
# from a presence/absence data table, i.e., a binary matrix of dimension (pun), where p
# is the population size (number of random variables) and n is the sample size (number of # data points)

##############################
# Step 0: Prepare the analysis

print("\nStep 0: Load modules and data and set options")

# Load the module for the inference with exact statistical tests
from holmes.data_analysis.exact_significative_interactions_1degree import *
import pandas as pd

# Choose the name of the directory (dir_name) where to save the files and the 'prefix' name of each
# created files (data_name)

dir_name = 'results'
data_name = 'small_data'

# Create target Directory if doesn't exist

if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    print("Directory ", dir_name, " Created ")
else:
    print("Directory ", dir_name, " already exists")

data_name = os.path.join(dir_name, data_name)

# Choose the significance level alpha to use throughout the analysis.
alpha = 0.01

# Number of samples that will generate our exact distribution (higher is better, but more time consuming)
nb_samples = 1000000

# Load data
data_matrix = np.load('sample_data.npy')
data_matrix = data_matrix.astype(np.int64)
p,n = data_matrix.shape

np.random.seed(55)
random_indices = np.random.choice(n, size=40, replace=False)
data_matrix = data_matrix[:, random_indices]
p,n = data_matrix.shape

print("Preparing to analyze a data set with " + str(p) + " variables and " + str(n) + " samples")

if n >= 10 and n <= 40:
    print('The number of samples is between 10 and 40. The CSV files can be used to fetch the pvalues\n'
          'Note that these pvalues were obtained using 1 000 000 samples for the exact distribution. \n'
          'If you wish to use a different number of samples to generate the exact distribution, change\n'
          'parameter « nb_samples », which will force your script to generate the new exact distributions.')

##############################
# First step: Extract all the unique tables

print("\nStep 1: Extract all the unique tables")

# Finds all unique tables
find_unique_tables(data_matrix, data_name)

print("Unique contingency tables saved in file " + data_name + "_table_list.json")

##############################
# Second step : Extract pvalues for all tables with an exact Chi1 distribution

print('Step 2: Extract pvalues for all tables with an exact Chi1 distribution')

if n >= 10 and n <= 40 and nb_samples == 1000000:
    df = pd.read_csv(r'exact_pvalues\clean_tablepvalues_withzeros_' + str(n) + '.csv')

    pvalues_for_tables_exact_with_df(data_name, df)
else:
    pvalues_for_tables_exact(data_name, nb_samples, data_matrix.shape[1])

print("\nResulting p-values saved in file " + data_name + "_exact_pval_dictio.json")

##############################
# Third step: Find table for all links and their associated pvalue

print('\nStep 3 : Find table for all links and their associated pvalue')

with open(data_name + '_exact_pval_dictio.json') as jsonfile:
    dictio = json.load(jsonfile)

    save_pairwise_p_values_phi_dictionary(data_matrix, dictio, data_name + '_exact_pvalues')

##############################
# Fourth step: Choose alpha and extract the network

print('\nStep 4: Generate network and extract edge_list for a given alpha')

g = read_pairwise_p_values(data_name + '_exact_pvalues.csv', alpha)

nx.write_edgelist(g, data_name + '_exact_edge_list_' + str(alpha)[2:] + '.txt', data=True)

print('Number of nodes : ', g.number_of_nodes())
print('Number of links : ', g.number_of_edges())
print("Edge list saved in file " + data_name + "_exact_edge_list_" + str(alpha)[2:] + ".txt")

##############################
# Fifth step: Find all triangles in the previous network

print('\nStep 5: Find all empty triangles in the network')

g = read_pairwise_p_values(data_name + '_exact_pvalues.csv', alpha)

save_all_triangles(g, data_name + '_exact_triangles_' + str(alpha)[2:])

print('Number of triangles : ', count_triangles_csv(data_name + '_exact_triangles_' + str(alpha)[2:] + '.csv'))

triangles_p_values_exact(data_name, nb_samples, data_name + '_exact_triangles_' + str(alpha)[2:] + '.csv', data_name + '_exact_triangles_' + str(alpha)[2:] + '_pvalues.csv', data_matrix)

##############################
# Sixth step: Extract all 2-simplices

print("\nStep 6: Extract all 2-simplices")

significant_triplet_from_csv(data_name + '_exact_triangles_' + str(alpha)[2:] + '_pvalues.csv', alpha, data_name + '_exact_2-simplices_' + str(alpha)[2:])

print("2-simplices saved in " + data_name + '_exact_2-simplices_' + str(alpha)[2:] + '.csv')