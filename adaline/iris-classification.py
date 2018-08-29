import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaline import AdalineGD
from plot_decision_regions import plot_decision_regions

# read in iris data set 
df = pd.read_csv('../iris-data.csv')

# select setosa and versicolor
y = df.iloc[0:99, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:99, [0,2]].values

# plot data
plt.scatter(X[:49,0], X[:49,1],
            color='red', marker='o', label='setosa')
plt.scatter(X[49:99,0], X[49:99,1],
            color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.xlabel('petal length [cm]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X,y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

plt.tight_layout()
plt.show()

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.xlabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
