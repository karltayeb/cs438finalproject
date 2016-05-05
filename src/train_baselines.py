from prep import *
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from evaluate import *
import pickle



data = Prep(logtransform=True, scaledata=True)

print "Predicting BRCA with linear SVM"
print '50:'
train_model(data, 50, 'topmixed', 'linear', 0, 'test')
test_metrics(data, 50, 'topmixed', 'test_svms.p', 'test_folds.p', 0)
print '100:'
train_model(data, 100, 'topmixed', 'linear', 0, 'test')
test_metrics(data, 100, 'topmixed', 'test_svms.p', 'test_folds.p', 0)
print '200:'
train_model(data, 200, 'topmixed', 'linear', 0, 'test')
test_metrics(data, 200, 'topmixed', 'test_svms.p', 'test_folds.p', 0)
print '300:'
train_model(data, 300, 'topmixed', 'linear', 0, 'test')
test_metrics(data, 300, 'topmixed', 'test_svms.p', 'test_folds.p', 0)
print '500:'
train_model(data, 500, 'topmixed', 'linear', 0, 'test')
test_metrics(data, 500, 'topmixed', 'test_svms.p', 'test_folds.p', 0)

print "\n\nPredicting PRAD with linear SVM"
print '50:'
train_model(data, 50, 'topmixed', 'linear', 1, 'test')
test_metrics(data, 50, 'topmixed', 'test_svms.p', 'test_folds.p', 1)
print '100:'
train_model(data, 100, 'topmixed', 'linear', 1, 'test')
test_metrics(data, 100, 'topmixed', 'test_svms.p', 'test_folds.p', 1)
print '200:'
train_model(data, 200, 'topmixed', 'linear', 1, 'test')
test_metrics(data, 200, 'topmixed', 'test_svms.p', 'test_folds.p', 1)
print '300:'
train_model(data, 300, 'topmixed', 'linear', 1, 'test')
test_metrics(data, 300, 'topmixed', 'test_svms.p', 'test_folds.p', 1)
print '500:'
train_model(data, 500, 'topmixed', 'linear', 1, 'test')
test_metrics(data, 500, 'topmixed', 'test_svms.p', 'test_folds.p', 1)

"""
print "\n\nPredicting BRCA with RBF kernel SVM"
print '100:'
train_model(data, 100, 'topmixed', 'rbf', 0, 'test')
test_metrics(data, 100, 'topmixed', 'test_svms.p', 'test_folds.p', 0)
print '200:'
train_model(data, 200, 'topmixed', 'rbf', 0, 'test')
test_metrics(data, 200, 'topmixed', 'test_svms.p', 'test_folds.p', 0)
print '300:'
train_model(data, 300, 'topmixed', 'rbf', 0, 'test')
test_metrics(data, 300, 'topmixed', 'test_svms.p', 'test_folds.p', 0)

print "\n\nPredicting PRAD with linear SVM"
print '100:'
train_model(data, 100, 'topmixed', 'rbf', 1, 'test')
test_metrics(data, 100, 'topmixed', 'test_svms.p', 'test_folds.p', 1)
print '200:'
train_model(data, 200, 'topmixed', 'rbf', 1, 'test')
test_metrics(data, 200, 'topmixed', 'test_svms.p', 'test_folds.p', 1)
print '300:'
train_model(data, 300, 'topmixed', 'rbf', 1, 'test')
test_metrics(data, 300, 'topmixed', 'test_svms.p', 'test_folds.p', 1)
"""


def train_model(prep, n, featureselect, ker, cancer_type, picklename):
	"""
	Train model learns the svm on all the training data (for plots)
	Trains model with 5-fold crossvalidation

	prep is a Prep object
	n is the number of features we want to use
	featureselect is the feature selection heuristic we use
	ker is the kernel of the model
	cancer_type is the cancer type: 0 for BRCA, 1 for PRAD
	picklename the base filename to save pickles as

	Saves:
	-- picklename_svm.p which is a trained svm on all training data
	-- picklename_folds.p which is a sklearn crossvalidation iterator
	-- picklename_svms.p the individual svms learned during cross validation
	"""

	train_expression = cancer_type * 4
	train_event = cancer_type * 4 + 2

	# select features
	features = prep.select_features(n, featureselect)
	expression_data = prep.dat[train_expression][:, features]
	event_data = prep.dat[train_event]

	svm = SVC(kernel=ker, C=1, probability=True)
	svm.fit(expression_data, event_data)
	savename = picklename + '_svm.p'
	pickle.dump(svm, open(savename, 'wb'))

	fivefold = KFold(data.dat[train_event].size, 5)

	# save folds so that we can test later
	savename = picklename + '_folds.p'
	pickle.dump(fivefold, open(savename, 'wb'))
	models = np.empty(5, dtype=object)

	i = 0
	for train, validate in fivefold:
		# select features on training set
		if cancer_type == 0:
			features = prep.select_features(n, featureselect, brca_indices=train, prad_indices=None)
		if cancer_type == 1:
			features = prep.select_features(n, featureselect, brca_indices=None, prad_indices=train)

		expression_data = prep.dat[train_expression][:, features]
		event_data = prep.dat[train_event]

		X_train, X_val = expression_data[train, :], expression_data[validate, :]
		y_train, y_val = event_data[train], event_data[validate]
		svm = SVC(kernel=ker, C=1, probability=True)
		svm.fit(X_train, y_train)
		models[i] = svm
		i += 1

	# save array of svms into pickle
	savename = picklename + '_svms.p'
	pickle.dump(models, open(savename, 'wb'))
