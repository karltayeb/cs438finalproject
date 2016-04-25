from prep import *
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn import cross_validation
import pickle


data = Prep(50)

fivefold = KFold(data.brca_event_train.size, 5)
# save folds so that we can test later
pickle.dump(fivefold, open('brca50_folds.p', 'wb'))
models = np.empty(5, dtype=object)
i = 0
for train, validate in fivefold:
	X_train, X_test = data.brca_expression_train[train, :], data.brca_expression_train[validate, :]
	y_train, y_test = data.brca_event_train[train], data.brca_event_train[validate]
	svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
	models[i] = svm
	i += 1
# save array of svms into pickle
pickle.dump(models, open('brca50_svms.p', 'wb'))

fivefold = KFold(data.prad_event_train.size, 5)
# save folds so that we can test later
pickle.dump(fivefold, open('prad50_folds.p', 'wb'))
models = np.empty(5, dtype=object)
i = 0
for train, validate in fivefold:
	X_train, X_test = data.prad_expression_train[train, :], data.prad_expression_train[validate, :]
	y_train, y_test = data.prad_event_train[train], data.prad_event_train[validate]
	svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
	models[i] = svm
	i += 1
# save array of svms into pickle
pickle.dump(models, open('prad50_svms.p', 'wb'))



data = Prep(100)

fivefold = KFold(data.brca_event_train.size, 5)
# save folds so that we can test later
pickle.dump(fivefold, open('brca100_folds.p', 'wb'))
models = np.empty(5, dtype=object)
i = 0
for train, validate in fivefold:
	X_train, X_test = data.brca_expression_train[train, :], data.brca_expression_train[validate, :]
	y_train, y_test = data.brca_event_train[train], data.brca_event_train[validate]
	svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
	models[i] = svm
	i += 1
# save array of svms into pickle
pickle.dump(models, open('brca100_svms.p', 'wb'))

fivefold = KFold(data.prad_event_train.size, 5)
# save folds so that we can test later
pickle.dump(fivefold, open('prad100_folds.p', 'wb'))
models = np.empty(5, dtype=object)
i = 0
for train, validate in fivefold:
	X_train, X_test = data.prad_expression_train[train, :], data.prad_expression_train[validate, :]
	y_train, y_test = data.prad_event_train[train], data.prad_event_train[validate]
	svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
	models[i] = svm
	i += 1
# save array of svms into pickle
pickle.dump(models, open('prad100_svms.p', 'wb'))



data = Prep(150)

fivefold = KFold(data.brca_event_train.size, 5)
# save folds so that we can test later
pickle.dump(fivefold, open('brca150_folds.p', 'wb'))
models = np.empty(5, dtype=object)
i = 0
for train, validate in fivefold:
	X_train, X_test = data.brca_expression_train[train, :], data.brca_expression_train[validate, :]
	y_train, y_test = data.brca_event_train[train], data.brca_event_train[validate]
	svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
	models[i] = svm
	i += 1
# save array of svms into pickle
pickle.dump(models, open('brca150_svms.p', 'wb'))

fivefold = KFold(data.prad_event_train.size, 5)
# save folds so that we can test later
pickle.dump(fivefold, open('prad150_folds.p', 'wb'))
models = np.empty(5, dtype=object)
i = 0
for train, validate in fivefold:
	X_train, X_test = data.prad_expression_train[train, :], data.prad_expression_train[validate, :]
	y_train, y_test = data.prad_event_train[train], data.prad_event_train[validate]
	svm = SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
	models[i] = svm
	i += 1
# save array of svms into pickle
pickle.dump(models, open('prad150_svms.p', 'wb'))