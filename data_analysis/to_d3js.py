
# TODO : This part of the code is used to transform the graph into something we could push to a D3JS notebook with
# Observable.

#### to json for d3js :

g = read_pairwise_p_values(data_name + '_asymptotic_pvalues.csv', alpha)

node_dictio_list = []
for noeud in g.nodes:
    node_dictio_list.append({"id": str(noeud), "group": 1})
    # node_dictio_list.append({"id":str(noeud)})

link_dictio_list = []
for lien in g.edges:
    link_dictio_list.append({"source": str(lien[0]), "target": str(lien[1]), "value": 1})

triplex_dictio_list = []

with open("TEST_TRIPLETS_ASYMPT.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        print(row)
        try:
            pval = float(row[-1])
            if pval < 0.001:
                triplex_dictio_list.append({"nodes": [str(row[0]), str(row[1]), str(row[2])]})
        except:
            pass


json_diction = {"nodes": node_dictio_list, "links": link_dictio_list, "triplex": triplex_dictio_list}
with open('d3js_simplicialcomplex_asympt_001.json', 'w') as outfile:
    json.dump(json_diction, outfile)
exit()
# Extract nodes with groups :
######groupe_set = set()
######with open('groupes_otu.csv', 'r') as csvfile:
######    reader = csv.reader(csvfile)
######    for row in reader:
######        try:
######            groupe_set.add(row[1])
######        except:
######            if row[0] != 'Bacteria':
######                groupe_set.add(row[0])
######print(len(groupe_set))
######print(groupe_set)
# node_dictio_list = []
# label_list = []
# with open('groupes_otu.csv', 'r') as csvfile:
#    reader = csv.reader(csvfile)
#    for row in reader:
#        try:
#            label_list.append(row[1])
#        except:
#            if row[0] != 'Bacteria':
#                label_list.append(row[0])
#            else:
#                label_list.append(label_list[-1])

############## D3JS NEGATIVE INTERACTIONS
# link_dictio_list = []
# node_set = set()
# count = 0
# for lien in g.edges:
#    if g.get_edge_data(lien[0], lien[1])['phi'] < 0 :
#        count += 1
#        #link_dictio_list.append({"source": str(lien[0]), "value": g.get_edge_data(lien[0], lien[1])['phi'], "target": str(lien[1])})
#        link_dictio_list.append({"source": str(lien[0]), "value": 1, "target": str(lien[1])})
#        node_set.add(lien[0])
#        node_set.add(lien[1])
# print('NUmber of negative interactions : ', count)

# node_dictio_list = []
# for noeud in list(node_set):
#    node_dictio_list.append({"id": str(noeud), "group": label_list[noeud]})
#    #node_dictio_list.append({"id": str(noeud)})

# json_diction = {"nodes": node_dictio_list, "links": link_dictio_list}
# with open('d3js_exact_chi1_negative_interactions_final_otu_with_groups.json', 'w') as outfile:
#    json.dump(json_diction, outfile)

# exit()
# link_dictio_list = []
# node_set = set()
# for lien in g.edges:
#    if g.get_edge_data(lien[0], lien[1])['phi'] > 0:
#        #link_dictio_list.append(
#        #    {"source": str(lien[0]), "value": g.get_edge_data(lien[0], lien[1])['phi'], "target": str(lien[1])})
#        link_dictio_list.append(
#            {"source": str(lien[0]), "value": 1, "target": str(lien[1])})
#        node_set.add(lien[0])
#        node_set.add(lien[1])

# node_dictio_list = []
# for noeud in list(node_set):
#    node_dictio_list.append({"id": str(noeud), "group": label_list[noeud]})
#    #node_dictio_list.append({"id": str(noeud)})

# json_diction = {"nodes": node_dictio_list, "links": link_dictio_list}
# with open('positive_interactions_otu_grouped.json', 'w') as outfile:
#    json.dump(json_diction, outfile)

# exit()

# Bunch of stats can be found here : https://networkx.github.io/documentation/stable/reference/functions.html
###### Compute a few interesting quantities
# print("Network density : ", nx.density(g))
# print("Is connected : ", nx.is_connected(g))
###print("Triadic closure : ", nx.transitivity(g))
# degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence
# print("Average degree : ", np.sum(np.array(degree_sequence))/len(degree_sequence))
# degreeCount = collections.Counter(degree_sequence)
# deg, cnt = zip(*degreeCount.items())


# TODO : This part of the code is used to transform the graph into something we could push to a D3JS notebook with
# Observable.

#### to json for d3js :

# g = read_pairwise_p_values('exact_chi1_pvalues_birds.csv', 0.01)
# label_list = []
# with open('groupes_otu.csv', 'r') as csvfile:
#   reader = csv.reader(csvfile)
#   for row in reader:
#       try:
#           label_list.append(row[1])
#       except:
#           if row[0] != 'Bacteria':
#               label_list.append(row[0])
#           else:
#               label_list.append(label_list[-1])

# node_dictio_list = []
# for noeud in g.nodes:
#   node_dictio_list.append({"id": str(noeud), "group": 2})
#   #node_dictio_list.append({"id":str(noeud)})

# link_dictio_list = []
# for lien in g.edges:
#   link_dictio_list.append({"source": str(lien[0]), "target": str(lien[1]), "value": 1})

# triplex_dictio_list = []

# with open('all_cube_pval_dictionary') as jsonfile:
#    dictio = json.load(jsonfile)

# matrix1 = np.loadtxt('incidenceMatrix.txt').T
# matrix1 = matrix1.astype(int)
# table_set = set()
# for two_simplex in tqdm(itertools.combinations(range(matrix1.shape[0]), 3)):

#    computed_cont_table = get_cont_cube(two_simplex[0], two_simplex[1], two_simplex[2], matrix1)
#    table_str = str(int(computed_cont_table[0, 0, 0])) + '_' + str(
#        int(computed_cont_table[0, 0, 1])) + '_' + str(int(computed_cont_table[0, 1, 0])) + '_' + str(
#        int(computed_cont_table[0, 1, 1])) + '_' + str(int(computed_cont_table[1, 0, 0])) + '_' + str(
#        int(computed_cont_table[1, 0, 1])) + '_' + str(int(computed_cont_table[1, 1, 0])) + '_' + str(
#        int(computed_cont_table[1, 1, 1]))
#    if table_str not in table_set:
#        table_set.add(table_str)
#    try:
#        if float(dictio[table_str][1]) < 0.01:
#            print(two_simplex[0], two_simplex[1], two_simplex[2], dictio[table_str][1])
#            triplex_dictio_list.append({"nodes": [str(two_simplex[0]), str(two_simplex[1]), str(two_simplex[2])]})
#            link_dictio_list.append({"source": str(two_simplex[0]), "target": str(two_simplex[1]), "value": 1})
#            link_dictio_list.append({"source": str(two_simplex[1]), "target": str(two_simplex[2]), "value": 1})
#            link_dictio_list.append({"source": str(two_simplex[0]), "target": str(two_simplex[2]), "value": 1})
#    except:
#        pass


# json_diction = {"nodes": node_dictio_list, "links" : link_dictio_list, "triplex" : triplex_dictio_list}
# with open('d3js_simplicialcomplex_01.json', 'w') as outfile:
#   json.dump(json_diction, outfile)
# exit()
# Extract nodes with groups :
######groupe_set = set()
######with open('groupes_otu.csv', 'r') as csvfile:
######    reader = csv.reader(csvfile)
######    for row in reader:
######        try:
######            groupe_set.add(row[1])
######        except:
######            if row[0] != 'Bacteria':
######                groupe_set.add(row[0])
######print(len(groupe_set))
######print(groupe_set)
# node_dictio_list = []
# label_list = []
# with open('groupes_otu.csv', 'r') as csvfile:
#    reader = csv.reader(csvfile)
#    for row in reader:
#        try:
#            label_list.append(row[1])
#        except:
#            if row[0] != 'Bacteria':
#                label_list.append(row[0])
#            else:
#                label_list.append(label_list[-1])

############## D3JS NEGATIVE INTERACTIONS
# link_dictio_list = []
# node_set = set()
# count = 0
# for lien in g.edges:
#    if g.get_edge_data(lien[0], lien[1])['phi'] < 0 :
#        count += 1
#        #link_dictio_list.append({"source": str(lien[0]), "value": g.get_edge_data(lien[0], lien[1])['phi'], "target": str(lien[1])})
#        link_dictio_list.append({"source": str(lien[0]), "value": 1, "target": str(lien[1])})
#        node_set.add(lien[0])
#        node_set.add(lien[1])
# print('NUmber of negative interactions : ', count)

# node_dictio_list = []
# for noeud in list(node_set):
#    node_dictio_list.append({"id": str(noeud), "group": label_list[noeud]})
#    #node_dictio_list.append({"id": str(noeud)})

# json_diction = {"nodes": node_dictio_list, "links": link_dictio_list}
# with open('d3js_exact_chi1_negative_interactions_final_otu_with_groups.json', 'w') as outfile:
#    json.dump(json_diction, outfile)

# exit()
# link_dictio_list = []
# node_set = set()
# for lien in g.edges:
#    if g.get_edge_data(lien[0], lien[1])['phi'] > 0:
#        #link_dictio_list.append(
#        #    {"source": str(lien[0]), "value": g.get_edge_data(lien[0], lien[1])['phi'], "target": str(lien[1])})
#        link_dictio_list.append(
#            {"source": str(lien[0]), "value": 1, "target": str(lien[1])})
#        node_set.add(lien[0])
#        node_set.add(lien[1])

# node_dictio_list = []
# for noeud in list(node_set):
#    node_dictio_list.append({"id": str(noeud), "group": label_list[noeud]})
#    #node_dictio_list.append({"id": str(noeud)})

# json_diction = {"nodes": node_dictio_list, "links": link_dictio_list}
# with open('positive_interactions_otu_grouped.json', 'w') as outfile:
#    json.dump(json_diction, outfile)

# exit()
