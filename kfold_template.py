from sklearn.model_selection import KFold
from sklearn import metrics

def run_kfold(machine, data, target, n, continuous):
	kfold_object = KFold(n_splits=n)
	#split the data into n parts
	kfold_object.get_n_splits(data)
	#stores n indexes to split the dataset

	i=0
	for train_index, test_index in kfold_object.split(data):
		i = i + 1
		print("Round:", str(i))
		print("training index: ")
		print(train_index)
		print("Testing Index: ")
		print(test_index)

		data_train = data[train_index]
		target_train = target[train_index]
		data_test = data[test_index]
		target_test = target[test_index]
		#machine = linear_model.LinearRegression()
		machine.fit(data_train, target_train)

		prediction = machine.predict(data_test)
		if (continuous==True):
			r2 = metrics.r2_score(target_test, prediction)
			print("R square score: ", r2)
			#continuous= R^2, dummy Y= accuracy score
			print("\n\n")
		else:
			accuracy_score = metrics.accuracy_score(target_test, prediction)
			print("accuracy_score: ", accuracy_score)

			confusion_matrix = metrics.confusion_matrix(target_test, prediction)
			print("confusion_matrix: ")
			print(confusion_matrix)
			#matrix is always in how many possible valuables you can have. First Column in 0, second 1. First number- the number of times the program is 
			#correct same with bottom right. Top right and bottom left are how many times it is wrong. This is all for a 0 or 1 result for Y. Accuracy score 
			#is successes divided by the total test pool. If Y can be 0,1,or2, then the matrix will be 3x3. All you have to do to get this is change the 
			#collumn we want intarget = dataset.iloc
			#continuous= R^2, dummy Y= accuracy score, confusion matrix
			print("\n\n")
