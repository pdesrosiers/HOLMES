##############################
# This short script provides an example of presence/absence data generated from a simplicial complex 

##############################
# Step 0: Prepare analysis
from tools import *

# Set parameters
observations = 1000
alpha = 0.01
build_sc = False

# Define the simplicial complex of interest
vp_1simp = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 5], [5, 6], [5, 7], [6, 7], [4, 8], [7, 8], [7, 9], [8, 9]]
vp_2simp = [[0, 1, 2], [1, 2, 3], [0, 1, 3], [0, 2, 3],  [7, 8, 9]]
facet_list = [[3, 4], [3, 5], [4, 5], [5, 6], [5, 7], [6, 7], [4, 8],[0, 1, 2], [1, 2, 3], [0, 1, 3], [0, 2, 3],  [7, 8, 9]]

##############################
# Step 1: Find a good factor graph

# The next loop finds a factor graph without induced interactions (interactions that we didn't specify)
# TODO Even though that's the goal of  this loop, I think there are missing steps. For instance, we should also
# check if all interactions in the factor graph are in vp_1simp and vp_2simp
i = 1
switch = True
while switch:
    print(i)
    factorgraph = FactorGraph(facet_list, N=observations, alpha=alpha, build_sc=build_sc)

    fg_1simplices_list, fg_2simplices_list = get_fg_vp_pathless(factorgraph, observations, alpha)

    #if len(fg_1simplices_list) == 16 and len(fg_2simplices_list) == 5:
    bad1simp = False
    bad2simp = False
    for onesimp in vp_1simp:
        if onesimp not in fg_1simplices_list:
            bad1simp = True
            break

    for twosimp in vp_2simp:
        if twosimp not in fg_2simplices_list:
            bad2simp = True
            break

    if not bad1simp and not bad2simp:
        switch = False
    i += 1
    if i>1000:
        print("Too long")
        switch = False
        
print(fg_1simplices_list)
print(fg_2simplices_list)

# Save factor graph
with open(r'Directory\fg.pkl', 'wb') as output:
    pickle.dump(factorgraph, output, pickle.HIGHEST_PROTOCOL)

with open(r'Directory\fg.pkl', 'rb') as fg_file:
    factorgraph = pickle.load(fg_file)
    probdist = Prob_dist(factorgraph)

    
##############################
# Step 2: Generated synthetic data  

#The next loop generates k Biadjacency matrices AKA presence/absence matrices.
#The variable observations gives the number of sites we want to generate for each matrix.       
k = 3
for i in np.arange(0, k, 1):
    state = np.random.randint(2, size=len(factorgraph.node_list))
    print(state)
    
    energy_obj = Energy(state, factorgraph)
    proposer = BitFlipProposer(factorgraph, energy_obj, state)
    sampler = Sampler(proposer, temperature=1, initial_burn=5, sample_burn=2**len(factorgraph.node_list))
    sampler.sample(observations)
    
    bipartite = build_bipartite(sampler.results['sample'])
    np.save(r'Directory\bipartite_' + str(i), bipartite)    
