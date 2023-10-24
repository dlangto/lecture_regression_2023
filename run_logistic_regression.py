import pandas
from sklearn import linear_model

dataset = pandas.read_csv("dataset.csv")

print(dataset)

target = dataset.iloc[:,1].values
# sets y values from first three columns
data = dataset.iloc[:,3:9].values
# sets x values from columns 3-8

machine = linear_model.LogisticRegression()
machine.fit(data,target)
#make a linear regression for x, y

new_dataset = pandas.read_csv("new_dataset.csv")
new_dataset = new_dataset.values
#import new dataset to make a prediction
prediction = machine.predict(new_dataset)

print(prediction)
#the three numbers are y predictions for the three people represented in new_dataset