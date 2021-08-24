from holmes.data_analysis.analyse_betti import *

data_name = 'opsahl-ucforum'
dir_name = 'results'
df = pd.read_csv('KONECT_data\out.' + data_name, sep=' ', skiprows=2, header=None)

k = pd.crosstab(df[0], df[1])

data_matrix = k.to_numpy()

pd_matrix = pd.DataFrame(data_matrix)
p, n = data_matrix.shape
print("Preparing to analyze a data set with " + str(p) + " variables and " + str(n) + " samples")

onesimp_file = os.path.join(dir_name, data_name + '_asymptotic_pvalues.csv')
twosimp_file = os.path.join(dir_name, data_name + '_asymptotic_2-simplices_01.csv')
threesimp_file = os.path.join(dir_name, data_name + '_3-simplices.csv')
build_facet_list(data_name + '_facetlist.txt', data_matrix, onesimp_file, twosimp_file, threesimp_file, 0.01)



path = data_name + '_facetlist.txt'
facetlist = []
with open(path, 'r') as file:
    for l in file:
        facetlist.append([int(x) for x in l.strip().split()])
somme = 0
maxlist =[]
for elem in facetlist:
    maxlist.append(len(elem))
    somme += len(elem)
print(somme/len(facetlist))
print(min(maxlist), max(maxlist))

st = build_simplex_tree(facetlist, 3)
compute_nb_simplices(st, 3)

print(compute_betti(facetlist, 4))

exit()

data_name = 'web_of_life'
dir_name = 'results'

data_matrix = np.loadtxt(open(r'M_PL_062.csv', 'r'), delimiter=',')

data_matrix = (data_matrix > 0) * 1

data_matrix = data_matrix.T

p, n = data_matrix.shape
print("Preparing to analyze a data set with " + str(p) + " variables and " + str(n) + " samples")

onesimp_file = os.path.join(dir_name, data_name + '_asymptotic_pvalues.csv')
twosimp_file = os.path.join(dir_name, data_name + '_asymptotic_2-simplices_01.csv')
threesimp_file = os.path.join(dir_name, data_name + '_3-simplices.csv')
build_facet_list(data_name + '_facetlist.txt', data_matrix, onesimp_file, twosimp_file, threesimp_file, 0.01)



path = data_name + '_facetlist.txt'
facetlist = []
with open(path, 'r') as file:
    for l in file:
        facetlist.append([int(x) for x in l.strip().split()])
somme = 0
maxlist =[]
for elem in facetlist:
    maxlist.append(len(elem))
    somme += len(elem)
print(somme/len(facetlist))
print(min(maxlist), max(maxlist))

st = build_simplex_tree(facetlist, 3)
compute_nb_simplices(st, 3)

print(compute_betti(facetlist, 4))

exit()