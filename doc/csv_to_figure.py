import json
import csv
from holmes.data_analysis.asymptotic_significative_interactions import *
import pandas as pd

#### to json for d3js :


data_name = 'edit-liwikibooks'
dir_name = 'results'
df = pd.read_csv('KONECT_data\out.' + data_name, sep='\t', skiprows=1, header=None)

g = df.groupby(0)[1].apply(list).reset_index()

k = pd.get_dummies(g[1].apply(pd.Series).stack()).sum(level=0)

data_matrix = k.to_numpy().T

p, n = data_matrix.shape

alpha = 0.01
g = read_pairwise_p_values(r'D:\Users\Xavier\Documents\HOLMES\HOLMES\doc\results\edit-liwikibooks_asymptotic_pvalues.csv', alpha)

node_dictio_list = []
for noeud in range(0, p):
    node_dictio_list.append({"id": str(noeud), "group": 1})
    # node_dictio_list.append({"id":str(noeud)})

link_dictio_list = []
for lien in g.edges:
    link_dictio_list.append({"source": str(lien[0]), "target": str(lien[1]), "value": 1})

triplex_dictio_list = []

with open(r"D:\Users\Xavier\Documents\HOLMES\HOLMES\doc\results\edit-liwikibooks_asymptotic_triangles_01_pvalues.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        print(row)
        try:
            pval = float(row[-1])
            if pval < alpha:
                triplex_dictio_list.append({"nodes": [str(row[0]), str(row[1]), str(row[2])]})
        except:
            pass

fourplex_dictio_list = []

with open(r"D:\Users\Xavier\Documents\HOLMES\HOLMES\doc\results\edit-liwikibooks_3-simplices.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        print(row)
        fourplex_dictio_list.append({"nodes": [str(row[0]), str(row[1]), str(row[2]), str(row[3])]})


json_diction = {"nodes": node_dictio_list, "links": link_dictio_list, "triplex": triplex_dictio_list, "fourplex": fourplex_dictio_list}
with open('d3js_' + data_name + '_simplicialcomplex_01.json', 'w') as outfile:
    json.dump(json_diction, outfile)
