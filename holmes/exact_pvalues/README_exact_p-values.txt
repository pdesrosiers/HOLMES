# Understanding the csv files #

Each line in the csv file represents the results of the exact and asymptotic 
test for independance for a given contingency table of size N (indicated in the name of the file).

For each line we can find five entries dubbed table_str, chi_stat, pval_exact ,pval_asympt and diff.

table_str is the contingency table flattened and transformed into a string. Each value represent an entry in the contingency table.
The values are separated by underscores. The first two values are elements [0, 0] and [0, 1], while the last two are elements [1,0] and [1,1] in a zero indexed array.

chi_stat is the chi^2 statistic computed with the table and it's expected table under the hypothesis of independence (not present in the csv file).
This value is independent of the test used, so wether the exact test or asymptotic test is used, this value does not change. 

pval_exact is the exact p-value obtained by generating the exact distribution of the statistics and computing the probability to draw a value equal or higher than
chi_stat using this distribution. The exact distribution was generated with 1 000 000 values using the procedure explained in the paper.

pval_asympt is the p-value obtained by using chi_stat and a chi^2 distribution of degree 1. 

diff is the absolute difference between pval_exact and pval_asympt.

# BE ADVISED #

Some pval_exact entries are 2 and their associated diff is also 2. This is not an error, but rather a code. If pval_exact = 2, it means that the procedure 
to compute the exact distribution failed because the expected table is identical to the table (hence a chi_stat of 0) and contains zeros. Such tables do not 
offer the possibility to generate tables that have positive entries where zeros are present in the original table. They are considered as structural zeros 
and not sampling zeros. As a result, we cannot generate the exact distribution without adding positive counts in places of zeros in the original table. However,
doing so would influence the exact distribution and defeat the purpose of using it, since we introduce some kind of bias towards a distribution that might, or not,
represent the actual distribution that has led to this specific table. 

For these tables, HOLMES thus always concludes in independence, favoring the reduction of type 1 error. 
