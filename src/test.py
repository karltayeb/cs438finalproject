from prep import *
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn import cross_validation
import pickle

def BRCAmixed50():
	data = Prep(50, logtransform=True, scaledata=True, featureselect='mixed')

	svm = SVC(kernel='linear', C=1, probability=True)
	svm.fit(data.brca_expression_train, data.brca_event_train)
	pickle.dump(svm, open('brca50_mixed_svm.p', 'wb'))

	fivefold = KFold(data.brca_event_train.size, 5)
	# save folds so that we can test later
	pickle.dump(fivefold, open('brca50_mixed_folds.p', 'wb'))
	models = np.empty(5, dtype=object)
	i = 0
	for train, validate in fivefold:
		X_train, X_test = data.brca_expression_train[train, :], data.brca_expression_train[validate, :]
		y_train, y_test = data.brca_event_train[train], data.brca_event_train[validate]
		svm = SVC(kernel='linear', C=1, probability=True)
		svm.fit(X_train, y_train)
		models[i] = svm
		i += 1
	# save array of svms into pickle
	pickle.dump(models, open('brca50_mixed_svms.p', 'wb'))

def PRADmixed50():
	data = Prep(50, logtransform=True, scaledata=True, featureselect='mixed')

	svm = SVC(kernel='linear', C=1, probability=True).fit(data.prad_expression_train, data.prad_event_train)
	pickle.dump(svm, open('prad50_mixed_svm.p', 'wb'))

	fivefold = KFold(data.prad_event_train.size, 5)
	# save folds so that we can test later
	pickle.dump(fivefold, open('prad50_mixed_folds.p', 'wb'))
	models = np.empty(5, dtype=object)
	i = 0
	for train, validate in fivefold:
		X_train, X_test = data.prad_expression_train[train, :], data.prad_expression_train[validate, :]
		y_train, y_test = data.prad_event_train[train], data.prad_event_train[validate]
		svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
		models[i] = svm
		i += 1
	# save array of svms into pickle
	pickle.dump(models, open('prad50_mixed_svms.p', 'wb'))

def BRCAmixed100():
	data = Prep(100, logtransform=True, scaledata=True)

	svm = SVC(kernel='linear', C=1, probability=True).fit(data.brca_expression_train, data.brca_event_train)
	pickle.dump(svm, open('brca100_mixed_svm.p', 'wb'))

	fivefold = KFold(data.brca_event_train.size, 5)
	# save folds so that we can test later
	pickle.dump(fivefold, open('brca100_mixed_folds.p', 'wb'))
	models = np.empty(5, dtype=object)
	i = 0
	for train, validate in fivefold:
		X_train, X_test = data.brca_expression_train[train, :], data.brca_expression_train[validate, :]
		y_train, y_test = data.brca_event_train[train], data.brca_event_train[validate]
		svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
		models[i] = svm
		i += 1
	# save array of svms into pickle
	pickle.dump(models, open('brca100_mixed_svms.p', 'wb'))
	return data

def PRADmixed100():
	data = Prep(100, logtransform=True, scaledata=True)
	svm = SVC(kernel='linear', C=1, probability=True).fit(data.prad_expression_train, data.prad_event_train)
	pickle.dump(svm, open('prad100_mixed_svm.p', 'wb'))

	fivefold = KFold(data.prad_event_train.size, 5)
	# save folds so that we can test later
	pickle.dump(fivefold, open('prad100_mixed_folds.p', 'wb'))
	models = np.empty(5, dtype=object)
	i = 0
	for train, validate in fivefold:
		X_train, X_test = data.prad_expression_train[train, :], data.prad_expression_train[validate, :]
		y_train, y_test = data.prad_event_train[train], data.prad_event_train[validate]
		svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
		models[i] = svm
		i += 1
	# save array of svms into pickle
	pickle.dump(models, open('prad100_mixed_svms.p', 'wb'))
	return data


def BRCAmixed150():
	data = Prep(150, logtransform=True, scaledata=True)

	svm = SVC(kernel='linear', C=1, probability=True).fit(data.brca_expression_train, data.brca_event_train)
	pickle.dump(svm, open('brca150_mixed_svm.p', 'wb'))

	fivefold = KFold(data.brca_event_train.size, 5)
	# save folds so that we can test later
	pickle.dump(fivefold, open('brca150_mixed_folds.p', 'wb'))
	models = np.empty(5, dtype=object)
	i = 0
	for train, validate in fivefold:
		X_train, X_test = data.brca_expression_train[train, :], data.brca_expression_train[validate, :]
		y_train, y_test = data.brca_event_train[train], data.brca_event_train[validate]
		svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
		models[i] = svm
		i += 1
	# save array of svms into pickle
	pickle.dump(models, open('brca150_mixed_svms.p', 'wb'))
	return data

def PRADmixed150():
	data = Prep(150, logtransform=True, scaledata=True)

	svm = SVC(kernel='linear', C=1, probability=True).fit(data.prad_expression_train, data.prad_event_train)
	pickle.dump(svm, open('prad150_mixed_svm.p', 'wb'))

	fivefold = KFold(data.prad_event_train.size, 5)
	# save folds so that we can test later
	pickle.dump(fivefold, open('prad150_mixed_folds.p', 'wb'))
	models = np.empty(5, dtype=object)
	i = 0
	for train, validate in fivefold:
		X_train, X_test = data.prad_expression_train[train, :], data.prad_expression_train[validate, :]
		y_train, y_test = data.prad_event_train[train], data.prad_event_train[validate]
		svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
		models[i] = svm
		i += 1
	# save array of svms into pickle
	pickle.dump(models, open('prad150_mixed_svms.p', 'wb'))
	return data

def seperate150():
	data = Prep(150, logtransform=True, scaledata=True, featureselect='seperate')

	svm = SVC(kernel='linear', C=1, probability=True).fit(data.brca_expression_train, data.brca_event_train)
	pickle.dump(svm, open('brca150_seperate_svm.p', 'wb'))

	fivefold = KFold(data.brca_event_train.size, 5)
	# save folds so that we can test later
	pickle.dump(fivefold, open('brca150_seperate_folds.p', 'wb'))
	models = np.empty(5, dtype=object)
	i = 0
	for train, validate in fivefold:
		X_train, X_test = data.brca_expression_train[train, :], data.brca_expression_train[validate, :]
		y_train, y_test = data.brca_event_train[train], data.brca_event_train[validate]
		svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
		models[i] = svm
		i += 1
	# save array of svms into pickle
	pickle.dump(models, open('brca150_seperate_svms.p', 'wb'))

	svm = SVC(kernel='linear', C=1, probability=True).fit(data.prad_expression_train, data.prad_event_train)
	pickle.dump(svm, open('prad150_seperate_svm.p', 'wb'))

	fivefold = KFold(data.prad_event_train.size, 5)
	# save folds so that we can test later
	pickle.dump(fivefold, open('prad150_seperate_folds.p', 'wb'))
	models = np.empty(5, dtype=object)
	i = 0
	for train, validate in fivefold:
		X_train, X_test = data.prad_expression_train[train, :], data.prad_expression_train[validate, :]
		y_train, y_test = data.prad_event_train[train], data.prad_event_train[validate]
		svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
		models[i] = svm
		i += 1
	# save array of svms into pickle
	pickle.dump(models, open('prad150_seperate_svms.p', 'wb'))

def BRCAseperaterbf150():
	data = Prep(150, logtransform=True, scaledata=True, featureselect='seperate')

	svm = SVC(kernel='rbf', C=1, probability=True).fit(data.brca_expression_train, data.brca_event_train)
	pickle.dump(svm, open('brca150_seperate_svm.p', 'wb'))

	fivefold = KFold(data.brca_event_train.size, 5)
	# save folds so that we can test later
	pickle.dump(fivefold, open('brca150_seperate_folds.p', 'wb'))
	models = np.empty(5, dtype=object)
	i = 0
	for train, validate in fivefold:
		X_train, X_test = data.brca_expression_train[train, :], data.brca_expression_train[validate, :]
		y_train, y_test = data.brca_event_train[train], data.brca_event_train[validate]
		svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
		models[i] = svm
		i += 1


