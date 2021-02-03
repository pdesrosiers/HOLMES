##############################
# This short script provides an example of presence/absence data generated from a simplicial complex 

##############################
# Step 0: Prepare analysis

# Load packages
from holmes.generative_model.tools import *
import os

# Set parameters
observations = 1000 # number of samples, i.e., number of data points or number of sites
alpha = 0.01 # significance level
build_sc = True # build a simplicial complex instead of a hypergraph
hardconstraint = True # Forces the simplicial complex to be identical to the specified facet list bellow.

# Choose the name of the directory (dir_name) where to save the files and the 'prefix' name of each 
# created files (data_name)
dir_name = 'synthetic_data'
data_name = 'data'

# Create target Directory if doesn't exist
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    print("Directory ", dir_name, " Created ")
else:
    print("Directory ", dir_name, " already exists")

data_name = os.path.join(dir_name, data_name)

##############################
# Step 1: Define the higher-order structure of interest

# Minimal list of simplices.  This defines a simplicial complex
facet_list = [[3, 4], [3, 5], [4, 5], [5, 6], [5, 7], [6, 7], [4, 8],[0, 1, 2], [1, 2, 3], [0, 1, 3], [0, 2, 3],  [7, 8, 9]]

##############################
# Step 2: Find a good factor graph

factorgraph = FactorGraph(facet_list, N=observations, alpha=alpha, build_sc=build_sc, hard_constraint=False)

# Save factor graph
with open(data_name + '_fg.pkl', 'wb') as output:
    pickle.dump(factorgraph, output, pickle.HIGHEST_PROTOCOL)

with open(data_name + '_fg.pkl', 'rb') as fg_file:
    factorgraph = pickle.load(fg_file)
    probdist = Prob_dist(factorgraph)

##############################
# Step 3: Generate synthetic data  

# The next loop generates k Biadjacency matrices AKA presence/absence matrices.
# The variable observations gives the number of sites we want to generate for each matrix.       
k = 2
for i in np.arange(0, k, 1):
    state = np.random.randint(2, size=len(factorgraph.node_list))
    print(state)
    
    energy_obj = Energy(state, factorgraph)
    proposer = BitFlipProposer(factorgraph, energy_obj, state)
    sampler = Sampler(proposer, temperature=1, initial_burn=5, sample_burn=2**len(factorgraph.node_list))
    sampler.sample(observations)
    
    bipartite = build_bipartite(sampler.results['sample'])
    np.save(data_name + '_bipartite_' + str(i), bipartite)    
