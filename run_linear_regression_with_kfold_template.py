#run something froma template
import pandas
from sklearn import linear_model

import kfold_template

dataset = pandas.read_csv("dataset.csv")

dataset = dataset.sample(frac=1)

target = dataset.iloc[:,1].values
data =  dataset.iloc[:,3:9].values

machine = linear_model.LogisticRegression()
kfold_template.run_kfold(machine, data, target, 4, False)

