import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('insurance.csv')

print(df_train.columns)
print(df_train['charges'].describe())

#Display of the normal distributioon
sns.distplot(df_train['charges']);

var = 'bmi'
data = pd.concat([df_train['charges'], df_train[var]], axis = 1)
data.plot.scatter(x = var, y = 'charges', ylim = (0, 70000))

var = 'children'
data = pd.concat([df_train['charges'], df_train[var]], axis = 1)
data.plot.scatter(x = var, y = 'charges', ylim = (0, 70000))

var = 'age'
data = pd.concat([df_train['charges'], df_train[var]], axis = 1)
data.plot.scatter(x = var, y = 'charges', ylim = (0, 70000))


corrmat = df_train.corr()
f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax = .8, square = True);

k = 4
cols = corrmat.nlargest(k, 'charges')['charges'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale = 1.25)
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10},
yticklabels = cols.values, xticklabels = cols.values)

sns.set()
cols = ['charges', 'bmi', 'age', 'children']
sns.pairplot(df_train[cols], size = 2.5)

#display skewdess, does the plot follow the red line? is there a way to upload this into an ait dashboad?
sns.distplot(df_train['charges'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['charges'], plot = plt)

df_train['charges'] = np.log(df_train['charges'])

sns.distplot(df_train['charges'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['charges'], plot = plt)

sns.distplot(df_train['bmi'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['bmi'], plot = plt)

sns.distplot(df_train['age'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['age'], plot = plt)

#check out plotly for more ideas. There are a lot of things that can be done. Also, there is a new competition on kaggle to check out
#Theoretical quantities and probability plot
#what can ne added to the plot to add more statistical value
plt.show()
