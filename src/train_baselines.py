from prep import *
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from evaluate import *
from sklearn import metrics
import pickle


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
	C_range = np.logspace(-2, 2, 5)
	best_C = C_range[0]
	best_score = 0
	for c in C_range:
		svm = SVC(kernel=ker, C=c, probability=True)
		svm.fit(expression_data, event_data)

		fivefold = KFold(data.dat[train_event].size, 5)
		score = np.empty(5);

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
			score = np.append(
				score,
				metrics.roc_auc_score(y_val, svm.predict_proba(X_val)[:,1])
				)
		if np.mean(score) > best_score:
			best_C = c

	svm = SVC(kernel=ker, C=best_C, probability=True)
	svm.fit(expression_data, event_data)
	savename = '../pickles/' + picklename + '_svm.p'
	pickle.dump(svm, open(savename, 'wb'))

	fivefold = KFold(data.dat[train_event].size, 5)

	# save folds so that we can test later
	savename = '../pickles/' + picklename + '_folds.p'
	pickle.dump(fivefold, open(savename, 'wb'))
	models = np.empty(5, dtype=object)
	score = np.empty(5)
	i=0
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
	savename = '../pickles/' + picklename + '_svms.p'
	pickle.dump(models, open(savename, 'wb'))
	print 'C: ', best_C



data = Prep(logtransform=True, scaledata=True)

results = np.array(('Model', 'Val Acc', 'Val Prec', 'Val Rec', 'Val ROC-AUC', 'Test Acc', 'Test Prec', 'Test Rec', 'Test ROC-AUC'))
results = results.reshape(-1,1)

print "Predicting BRCA with BRCA selected features"
train_model(data, 50, 'brca', 'linear', 0, 'brcabrca50')
results = np.concatenate((results, test_metrics(data, 50, 'brca', 'brcabrca50_svms.p', 'brcabrca50_folds.p', 0, 'linear')), axis=-1)
ROC(data, 50, 'brca', 'brcabrca50_svms.p', 'brcabrca50_folds.p', 0, 'linear')

print '100:'
train_model(data, 100, 'brca', 'linear', 0, 'brcabrca100')
results = np.concatenate((results, test_metrics(data, 100, 'brca', 'brcabrca100_svms.p', 'brcabrca100_folds.p', 0, 'linear')), axis=-1)
ROC(data, 100, 'brca', 'brcabrca100_svms.p', 'brcabrca100_folds.p', 0, 'linear')

print '200:'
train_model(data, 200, 'brca', 'linear', 0, 'brcabrca200')
results = np.concatenate((results, test_metrics(data, 200, 'brca', 'brcabrca200_svms.p', 'brcabrca200_folds.p', 0, 'linear')), axis=-1)
ROC(data, 200, 'brca', 'brcabrca200_svms.p', 'brcabrca200_folds.p', 0, 'linear')

print '300:'
train_model(data, 300, 'brca', 'linear', 0, 'brcabrca300')
results = np.concatenate((results, test_metrics(data, 300, 'brca', 'brcabrca300_svms.p', 'brcabrca300_folds.p', 0, 'linear')), axis=-1)
ROC(data, 300, 'brca', 'brcabrca300_svms.p', 'brcabrca300_folds.p', 0, 'linear')

print '500:'
train_model(data, 500, 'brca', 'linear', 0, 'brcabrca500')
results = np.concatenate((results, test_metrics(data, 500, 'brca', 'brcabrca500_svms.p', 'brcabrca500_folds.p', 0, 'linear')), axis=-1)
ROC(data, 500, 'brca', 'brcabrca500_svms.p', 'brcabrca500_folds.p', 0, 'linear')


print "\n\nPredicting PRAD with PRAD selected features"
print '50:'
train_model(data, 50, 'prad', 'linear', 1, 'pradprad50')
results = np.concatenate((results, test_metrics(data, 50, 'prad', 'pradprad50_svms.p', 'pradprad50_folds.p', 1, 'linear')), axis=-1)
ROC(data, 50, 'prad', 'pradprad50_svms.p', 'pradprad50_folds.p', 1, 'linear')

print '100:'
train_model(data, 100, 'prad', 'linear', 1, 'pradprad100')
results = np.concatenate((results, test_metrics(data, 100, 'prad', 'pradprad100_svms.p', 'pradprad100_folds.p', 1, 'linear')), axis=-1)
ROC(data, 100, 'prad', 'pradprad100_svms.p', 'pradprad100_folds.p', 1, 'linear')

print '200:'
train_model(data, 200, 'prad', 'linear', 1, 'pradprad200')
results = np.concatenate((results, test_metrics(data, 200, 'prad', 'pradprad200_svms.p', 'pradprad200_folds.p', 1, 'linear')), axis=-1)
ROC(data, 200, 'prad', 'pradprad200_svms.p', 'pradprad200_folds.p', 1, 'linear')

print '300:'
train_model(data, 300, 'prad', 'linear', 1, 'pradprad300')
results = np.concatenate((results, test_metrics(data, 300, 'prad', 'pradprad300_svms.p', 'pradprad300_folds.p', 1, 'linear')), axis=-1)
ROC(data, 300, 'prad', 'pradprad300_svms.p', 'pradprad300_folds.p', 1, 'linear')

print '500:'
train_model(data, 500, 'prad', 'linear', 1, 'pradprad500')
results = np.concatenate((results, test_metrics(data, 500, 'prad', 'pradprad500_svms.p', 'pradprad500_folds.p', 1, 'linear')), axis=-1)
ROC(data, 500, 'prad', 'pradprad500_svms.p', 'pradprad500_folds.p', 1, 'linear')



print "\n\nPredicting BRCA with top mixed feature selection"
print '50:'
train_model(data, 50, 'topmixed', 'linear', 0, 'brcatopmixed50')
results = np.concatenate((results, test_metrics(data, 50, 'topmixed', 'brcatopmixed50_svms.p', 'brcatopmixed50_folds.p', 0, 'linear')), axis=-1)
ROC(data, 50, 'topmixed', 'brcatopmixed50_svms.p', 'brcatopmixed50_folds.p', 0, 'linear')

print '100:'
train_model(data, 100, 'topmixed', 'linear', 0, 'brcatopmixed100')
results = np.concatenate((results, test_metrics(data, 100, 'topmixed', 'brcatopmixed100_svms.p', 'brcatopmixed100_folds.p', 0, 'linear')), axis=-1)
ROC(data, 100, 'topmixed', 'brcatopmixed100_svms.p', 'brcatopmixed100_folds.p', 0, 'linear')

print '200:'
train_model(data, 200, 'topmixed', 'linear', 0, 'brcatopmixed200')
results = np.concatenate((results, test_metrics(data, 200, 'topmixed', 'brcatopmixed200_svms.p', 'brcatopmixed200_folds.p', 0, 'linear')), axis=-1)
ROC(data, 200, 'topmixed', 'brcatopmixed200_svms.p', 'brcatopmixed200_folds.p', 0, 'linear')

print '300:'
train_model(data, 300, 'topmixed', 'linear', 0, 'brcatopmixed300')
results = np.concatenate((results, test_metrics(data, 300, 'topmixed', 'brcatopmixed300_svms.p', 'brcatopmixed300_folds.p', 0, 'linear')), axis=-1)
ROC(data, 300, 'topmixed', 'brcatopmixed300_svms.p', 'brcatopmixed300_folds.p', 0, 'linear')

print '500:'
train_model(data, 500, 'topmixed', 'linear', 0, 'brcatopmixed500')
results = np.concatenate((results, test_metrics(data, 500, 'topmixed', 'brcatopmixed500_svms.p', 'brcatopmixed500_folds.p', 0, 'linear')), axis=-1)
ROC(data, 500, 'topmixed', 'brcatopmixed500_svms.p', 'brcatopmixed500_folds.p', 0, 'linear')



print "\n\nPredicting PRAD with top mixed feature selection"
print '50:'
train_model(data, 50, 'topmixed', 'linear', 1, 'pradtopmixed50')
results = np.concatenate((results, test_metrics(data, 50, 'topmixed', 'pradtopmixed50_svms.p', 'pradtopmixed50_folds.p', 1, 'linear')), axis=-1)
ROC(data, 50, 'topmixed', 'pradtopmixed50_svms.p', 'pradtopmixed50_folds.p', 1, 'linear')

print '100:'
train_model(data, 100, 'topmixed', 'linear', 1, 'pradtopmixed100')
results = np.concatenate((results, test_metrics(data, 100, 'topmixed', 'pradtopmixed100_svms.p', 'pradtopmixed100_folds.p', 1, 'linear')), axis=-1)
ROC(data, 100, 'topmixed', 'pradtopmixed100_svms.p', 'pradtopmixed100_folds.p', 1, 'linear')

print '200:'
train_model(data, 200, 'topmixed', 'linear', 1, 'pradtopmixed200')
results = np.concatenate((results, test_metrics(data, 200, 'topmixed', 'pradtopmixed200_svms.p', 'pradtopmixed200_folds.p', 1, 'linear')), axis=-1)
ROC(data, 200, 'topmixed', 'pradtopmixed200_svms.p', 'pradtopmixed200_folds.p', 1, 'linear')

print '300:'
train_model(data, 300, 'topmixed', 'linear', 1, 'pradtopmixed300')
results = np.concatenate((results, test_metrics(data, 300, 'topmixed', 'pradtopmixed300_svms.p', 'pradtopmixed300_folds.p', 1, 'linear')), axis=-1)
ROC(data, 300, 'topmixed', 'pradtopmixed300_svms.p', 'pradtopmixed300_folds.p', 1, 'linear')

print '500:'
train_model(data, 500, 'topmixed', 'linear', 1, 'pradtopmixed500')
results = np.concatenate((results, test_metrics(data, 500, 'topmixed', 'pradtopmixed500_svms.p', 'pradtopmixed500_folds.p', 1, 'linear')), axis=-1)
ROC(data, 500, 'topmixed', 'pradtopmixed500_svms.p', 'pradtopmixed500_folds.p', 1, 'linear')




print "\n\nPredicting BRCA with mixed feature selection"
print '50:'
train_model(data, 50, 'mixed', 'linear', 0, 'brcamixed50')
results = np.concatenate((results, test_metrics(data, 50, 'mixed', 'brcamixed50_svms.p', 'brcamixed50_folds.p', 0, 'linear')), axis=-1)
ROC(data, 50, 'mixed', 'brcamixed50_svms.p', 'brcamixed50_folds.p', 0, 'linear')

print '100:'
train_model(data, 100, 'mixed', 'linear', 0, 'brcamixed100')
results = np.concatenate((results, test_metrics(data, 100, 'mixed', 'brcamixed100_svms.p', 'brcamixed100_folds.p', 0, 'linear')), axis=-1)
ROC(data, 100, 'mixed', 'brcamixed100_svms.p', 'brcamixed100_folds.p', 0, 'linear')

print '200:'
train_model(data, 200, 'mixed', 'linear', 0, 'brcamixed200')
results = np.concatenate((results, test_metrics(data, 200, 'mixed', 'brcamixed200_svms.p', 'brcamixed200_folds.p', 0, 'linear')), axis=-1)
ROC(data, 200, 'mixed', 'brcamixed200_svms.p', 'brcamixed200_folds.p', 0, 'linear')

print '300:'
train_model(data, 300, 'mixed', 'linear', 0, 'brcamixed300')
results = np.concatenate((results, test_metrics(data, 300, 'mixed', 'brcamixed300_svms.p', 'brcamixed300_folds.p', 0, 'linear')), axis=-1)
ROC(data, 300, 'mixed', 'brcamixed300_svms.p', 'brcamixed300_folds.p', 0, 'linear')

print '500:'
train_model(data, 500, 'mixed', 'linear', 0, 'brcamixed500')
results = np.concatenate((results, test_metrics(data, 500, 'mixed', 'brcamixed500_svms.p', 'brcamixed500_folds.p', 0, 'linear')), axis=-1)
ROC(data, 500, 'mixed', 'brcamixed500_svms.p', 'brcamixed500_folds.p', 0, 'linear')




print "\n\nPredicting PRAD with mixed feature selection"
print '50:'
train_model(data, 50, 'mixed', 'linear', 1, 'pradmixed50')
results = np.concatenate((results, test_metrics(data, 50, 'mixed', 'pradmixed50_svms.p', 'pradmixed50_folds.p', 1, 'linear')), axis=-1)
ROC(data, 50, 'mixed', 'pradmixed50_svms.p', 'pradmixed50_folds.p', 1, 'linear')

print '100:'
train_model(data, 100, 'mixed', 'linear', 1, 'pradmixed100')
results = np.concatenate((results, test_metrics(data, 100, 'mixed', 'pradmixed100_svms.p', 'pradmixed100_folds.p', 1, 'linear')), axis=-1)
ROC(data, 100, 'mixed', 'pradmixed100_svms.p', 'pradmixed100_folds.p', 1, 'linear')

print '200:'
train_model(data, 200, 'mixed', 'linear', 1, 'pradmixed200')
results = np.concatenate((results, test_metrics(data, 200, 'mixed', 'pradmixed200_svms.p', 'pradmixed200_folds.p', 1, 'linear')), axis=-1)
ROC(data, 200, 'mixed', 'pradmixed200_svms.p', 'pradmixed200_folds.p', 1, 'linear')

print '300:'
train_model(data, 300, 'mixed', 'linear', 1, 'pradmixed300')
results = np.concatenate((results, test_metrics(data, 300, 'mixed', 'pradmixed300_svms.p', 'pradmixed300_folds.p', 1, 'linear')), axis=-1)
ROC(data, 300, 'mixed', 'pradmixed300_svms.p', 'pradmixed300_folds.p', 1, 'linear')

print '500:'
train_model(data, 500, 'mixed', 'linear', 1, 'pradmixed500')
results = np.concatenate((results, test_metrics(data, 500, 'mixed', 'pradmixed500_svms.p', 'pradmixed500_folds.p', 1, 'linear')), axis=-1)
ROC(data, 500, 'mixed', 'pradmixed500_svms.p', 'pradmixed500_folds.p', 1, 'linear')


np.savetxt('../out/baseline_results.txt', results, delimiter='\t', newline='\n', fmt='%s')


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



