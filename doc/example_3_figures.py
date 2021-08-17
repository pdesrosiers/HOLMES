import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

with open(r'sc_20_nodes\100\dictionary_of_truth.pkl', 'rb') as dot_file:
   dot_100 = pickle.load(dot_file)

with open(r'sc_20_nodes\1000\dictionary_of_truth.pkl', 'rb') as dot_file:
   dot_1000 = pickle.load(dot_file)

with open(r'sc_20_nodes\10000\dictionary_of_truth.pkl', 'rb') as dot_file:
   dot_10000 = pickle.load(dot_file)

### For 100 observations ### :

vp_one_100 = dot_100['vp_one']
fp_one_100 = dot_100['fp_one']

vp_two_100 = dot_100['vp_two']
fp_two_100 = dot_100['fp_two']

vp_one_count_100 = []
fp_one_count_100 = []
for vp_set in vp_one_100:
    vp_one_count_100.append(len(vp_set))

for fp_set in fp_one_100:
    fp_one_count_100.append(len(fp_set))

vp_two_count_100 = []
fp_two_count_100 = []
for vp_set in vp_two_100:
    vp_two_count_100.append(len(vp_set))

for fp_set in fp_two_100:
    fp_two_count_100.append(len(fp_set))



### For 1000 observations ### :

vp_one_1000 = dot_1000['vp_one']
fp_one_1000 = dot_1000['fp_one']

vp_two_1000 = dot_1000['vp_two']
fp_two_1000 = dot_1000['fp_two']

vp_one_count_1000 = []
fp_one_count_1000 = []
for vp_set in vp_one_1000:
    vp_one_count_1000.append(len(vp_set))

for fp_set in fp_one_1000:
    fp_one_count_1000.append(len(fp_set))

vp_two_count_1000 = []
fp_two_count_1000 = []
for vp_set in vp_two_1000:
    vp_two_count_1000.append(len(vp_set))

for fp_set in fp_two_1000:
    fp_two_count_1000.append(len(fp_set))


### For 10000 observations ### :

vp_one_10000 = dot_10000['vp_one']
fp_one_10000 = dot_10000['fp_one']

vp_two_10000 = dot_10000['vp_two']
fp_two_10000 = dot_10000['fp_two']

vp_one_count_10000 = []
fp_one_count_10000 = []
for vp_set in vp_one_10000:
    vp_one_count_10000.append(len(vp_set))

for fp_set in fp_one_10000:
    fp_one_count_10000.append(len(fp_set))

vp_two_count_10000 = []
fp_two_count_10000 = []
for vp_set in vp_two_10000:
    vp_two_count_10000.append(len(vp_set))

for fp_set in fp_two_10000:
    fp_two_count_10000.append(len(fp_set))




#df = pd.DataFrame(list(zip([vp_two_count_100 + vp_two_count_1000 + vp_two_count_10000], [[100]*100 + [1000]*100 + [10000]*100])), ['vp_two_count', 'nb_observations']).T



df = pd.DataFrame(list(zip(vp_two_count_100 + vp_two_count_1000 +vp_two_count_10000, [100]*100 + [1000]*100 + [10000]*100)) , columns = ['vp_two_count', 'nb_observations'])

#sns.violinplot(y=df["vp_two_count"], x=df["nb_observations"], inner='stick')

#sns.catplot(x="nb_observations", y="vp_two_count", kind="violin", inner=None, data=df)
#sns.swarmplot(x=df["nb_observations"], y=df["vp_two_count"], color="k", size=3)

#g = sns.catplot(x="nb_observations", y="vp_two_count", kind="violin", inner=None, data=df, scale='width')

#g = sns.catplot(x="nb_observations", y="vp_two_count", kind="box", data=df)
#sns.swarmplot(x="nb_observations", y="vp_two_count", color="k", size=3, data=df, ax=g.ax)

#g.set(xlabel='Number of observations / matrix ', ylabel='True positives 2-simplices (target = 7)')
#plt.show()

#exit()



def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Number of observations / matrix ')

plt.style.use('seaborn')



fig, ax1 = plt.subplots(nrows=1, ncols=1)

ax1.set_ylabel('Number of Inferred Links')
parts = ax1.violinplot([vp_one_count_100 , vp_one_count_1000, vp_one_count_10000], positions = [1, 4, 7], showmeans=False, showextrema=False, bw_method=0.12, widths=1.5)

for pc in parts['bodies']:
    pc.set_facecolor('#ff7f00ff')
    pc.set_alpha(1)

parts = ax1.violinplot([ fp_one_count_100, fp_one_count_1000, fp_one_count_10000], positions = [2, 5, 8], showmeans=False, showextrema=False, bw_method=0.12, widths=1.5)

for pc in parts['bodies']:
    pc.set_facecolor('#00a1ffff')
    pc.set_alpha(1)

#set_axis_style(ax1, labels=['100', '1000', '10 000'])
ax1.set_xticklabels(['100 observations/matrix', '1000 observations/matrix', '10000 observations/matrix'])
ax1.set_xticks([1.5, 4.5, 7.5])

labels = [(mpatches.Patch(color='#ff7f00ff'), 'True Positives (target = 32)'), (mpatches.Patch(color='#00a1ffff'), 'False Positives (target = 0)')]

plt.legend(*zip(*labels), loc=2)

plt.show()


fig2, ax2 = plt.subplots(nrows=1, ncols=1)

ax2.set_ylabel('Number of Inferred 2-simplices')
parts = ax2.violinplot([vp_two_count_100 , vp_two_count_1000, vp_two_count_10000], positions = [1, 4, 7], showmeans=False, showextrema=False, bw_method=0.12, widths=1.5)

for pc in parts['bodies']:
    pc.set_facecolor('#ff7f00ff')
    pc.set_alpha(1)

parts = ax2.violinplot([fp_two_count_100, fp_two_count_1000, fp_two_count_10000], positions = [2, 5, 8], showmeans=False, showextrema=False, bw_method=0.6, widths=1.5)

for pc in parts['bodies']:
    pc.set_facecolor('#00a1ffff')
    pc.set_alpha(1)

#set_axis_style(ax1, labels=['100', '1000', '10 000'])
ax2.set_xticklabels(['100 observations/matrix', '1000 observations/matrix', '10000 observations/matrix'])
ax2.set_xticks([1.5, 4.5, 7.5])

labels = [(mpatches.Patch(color='#ff7f00ff'), 'True Positives (target = 7)'), (mpatches.Patch(color='#00a1ffff'), 'False Positives (target = 0)')]

plt.legend(*zip(*labels), loc=2)

plt.show()

exit()



####### OLD CODE ##########

ax1.set_ylabel('True positive counts for links (target = 32)')
ax1.violinplot([vp_one_count_100 , fp_one_count_100], positions = [1, 2], showmeans=True, bw_method=0.12)

ax1.violinplot([vp_one_count_1000 , fp_one_count_1000], positions = [4, 5], showmeans=True, bw_method=0.12)

ax1.violinplot([vp_one_count_10000 , fp_one_count_10000], positions = [7, 8], showmeans=True, bw_method=0.12)

#set_axis_style(ax1, labels=['100', '1000', '10 000'])
ax1.set_xticklabels(['100 ', '1000', '10000'])
ax1.set_xticks([1.5, 4.5, 7.5])

plt.show()


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
#ax1.set_title('True positive counts for ')
#ax1.set_ylabel('True positive counts for links (target = 32)')
#ax1.violinplot([vp_one_count_100 ,vp_one_count_1000, vp_one_count_10000], showmeans=True, bw_method=0.12)
#set_axis_style(ax1, labels=['100', '1000', '10 000'])

#ax1.violinplot([fp_one_count_100, fp_one_count_1000, fp_one_count_10000], showmeans=True, bw_method=0.12)
#set_axis_style(ax1, labels=['100', '1000', '10 000'])

#ax2.set_title('Default violin plot')
ax2.set_ylabel('False positive counts for links (target = 0)')
ax2.violinplot([fp_one_count_100, fp_one_count_1000, fp_one_count_10000], showmeans=True, bw_method=0.12)
set_axis_style(ax2, labels=['100', '1000', '10 000'])

fig2, (ax3, ax4) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=False)
#ax3.set_title('True positive counts for ')
ax3.set_ylabel('True positive counts for 2-simplices (target = 7)')
ax3.violinplot([vp_two_count_100, vp_two_count_1000, vp_two_count_10000], showmeans=True, bw_method=0.12)
set_axis_style(ax3, labels=['100', '1000', '10 000'])

#ax4.set_title('Default violin plot')
ax4.set_ylabel('False positive counts for 2-simplices (target = 0)')
ax4.violinplot([fp_two_count_100, fp_two_count_1000, fp_two_count_10000], showmeans=True, bw_method=0.12)
set_axis_style(ax4, labels=['100', '1000', '10 000'])

plt.show()



#g = sns.catplot(x="Number of observations", y="True positive count", kind="violin", inner=None, data=vp_two_count_10000)
#sns.swarmplot(x="Number of observations", y="True positive count", color="k", size=3, data=vp_two_count_10000, ax=g.ax)
