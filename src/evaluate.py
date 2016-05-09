from prep import *
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc

def test_metrics(prep, n, featureselect, model_pickle, kfold_pickle, cancer_type, kernel):

	model = pickle.load(open('../pickles/'+ model_pickle, 'rb'))
	folds = pickle.load(open('../pickles/'+ kfold_pickle, 'rb'))

	train_expression = cancer_type * 4
	test_expression = cancer_type * 4 + 1
	train_event = cancer_type * 4 + 2
	test_event = cancer_type * 4 + 3

	validation_accuracy = np.empty(0)
	validation_precision = np.empty(0)
	validation_recall = np.empty(0)
	validation_rocauc = np.empty(0)
	test_accuracy = np.empty(0)
	test_precision = np.empty(0)
	test_recall = np.empty(0)
	test_rocauc = np.empty(0)

	cumulative_prediction = np.zeros(prep.dat[test_event].size)


	i = 0
	for train, validate in folds:

		if cancer_type == 0:
			features = prep.select_features(n, featureselect, brca_indices=train, prad_indices=None)
		if cancer_type == 1:
			features = prep.select_features(n, featureselect, brca_indices=None, prad_indices=train)

		train_expression_data = prep.dat[train_expression][:, features]
		train_event_data = prep.dat[train_event]
		test_expression_data = prep.dat[test_expression][:, features]
		test_event_data = prep.dat[test_event]

		validation_accuracy = np.append(
			validation_accuracy,
			metrics.accuracy_score(train_event_data[validate], model[i].predict(train_expression_data[validate,:]))
			)
		validation_precision = np.append(
			validation_precision,
			metrics.average_precision_score(train_event_data[validate], model[i].predict(train_expression_data[validate,:]))
			)
		validation_recall = np.append(
			validation_recall,
			metrics.recall_score(train_event_data[validate], model[i].predict(train_expression_data[validate,:]))
			)
		validation_rocauc = np.append(
			validation_rocauc,
			metrics.roc_auc_score(train_event_data[validate], model[i].predict_proba(train_expression_data[validate,:])[:,1])
			)
		test_accuracy = np.append(
			test_accuracy,
			metrics.accuracy_score(test_event_data, model[i].predict(test_expression_data))
			)
		test_precision = np.append(
			test_precision,
			metrics.average_precision_score(test_event_data, model[i].predict(test_expression_data))
			)
		test_recall = np.append(
			test_recall,
			metrics.recall_score(test_event_data, model[i].predict(test_expression_data))
			)
		test_rocauc = np.append(
			test_rocauc,
			metrics.roc_auc_score(test_event_data, model[i].predict_proba(test_expression_data)[:,1])
			)
		cumulative_prediction = cumulative_prediction + model[i].predict(test_expression_data)

	print 'Average results of 5-fold cross validated models:'
	print 'Mean validation set accuracy: ', np.mean(validation_accuracy)
	print 'Mean validation set precision: ', np.mean(validation_precision)
	print 'Mean validation set recall: ', np.mean(validation_recall)
	print 'Mean validation set ROC AUC score: ', np.mean(validation_rocauc)
	print 'Mean test set accuracy: ', np.mean(test_accuracy)
	print 'Mean test set precision: ', np.mean(test_precision)
	print 'Mean test set recall: ', np.mean(test_recall)
	print 'Mean test set ROC AUC score: ', np.mean(test_rocauc)

	cumulative_prediction = cumulative_prediction / 5
	cumulative_prediction[cumulative_prediction >= 0.5] = 1
	cumulative_prediction[cumulative_prediction < 0.5] = 0

	#print cumulative_prediction

	tn = np.logical_and(cumulative_prediction == 0, prep.dat[test_event] == 0)  # true positive
	fn = np.logical_and(cumulative_prediction == 0, prep.dat[test_event] == 1)  # false negative
	tp = np.logical_and(cumulative_prediction == 1, prep.dat[test_event] == 1)  # true positives
	fp = np.logical_and(cumulative_prediction == 1, prep.dat[test_event] == 0)  # false positives
	"""
	pca = PCA(2)
	projected_data = pca.fit(data[test_expression]).transform(data[test_expression])

	colors = np.random.rand(50)
	fig, ax = plt.subplots(ncols=1)
	#ax.scatter(projected_data[tp,0], projected_data[tp,1], c=colors, s=100, label='True Pos')
	ax.scatter(projected_data[tn,0], projected_data[tn,1], c=colors, s=100, label='True Neg')
	ax.scatter(projected_data[fp,0], projected_data[fp,1], c=colors, s=100, label='False Pos')
	ax.scatter(projected_data[fn,0], projected_data[fn,1], c=colors, s=100, label='False Neg')

	ax.set_xlabel("PC-1 of Test Data")
	ax.set_ylabel("PC-2 of Test Data")
	ax.set_title("Classification Results over PC1-PC2")

	plt.show()
	"""

	name = 'brca_'
	if cancer_type is 1:
		name = 'prad_'
	name = name + featureselect + '_' + str(n) + '_' + kernel

	out = np.array((name, str(np.mean(validation_accuracy)),
		str(np.mean(validation_precision)),
		str(np.mean(validation_recall)),
		str(np.mean(validation_rocauc)),
		str(np.mean(test_accuracy)),
		str(np.mean(test_precision)),
		str(np.mean(test_recall)),
		str(np.mean(test_rocauc))))

	out = out.reshape(-1,1)
	return out
	#return projected_data, tp

def ROC(prep, n, featureselect, model_pickle, kfold_pickle, cancer_type, kernel):

	name = 'brca_'
	if cancer_type is 1:
		name = 'prad_'
	name = name + featureselect + '_' + str(n) + '_' + kernel

	model = pickle.load(open('../pickles/'+ model_pickle, 'rb'))
	folds = pickle.load(open('../pickles/'+ kfold_pickle, 'rb'))

	train_expression = cancer_type * 4
	test_expression = cancer_type * 4 + 1
	train_event = cancer_type * 4 + 2
	test_event = cancer_type * 4 + 3

	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	all_tpr = []

	for i, (train, test) in enumerate(folds):

		if cancer_type == 0:
			features = prep.select_features(n, featureselect, brca_indices=train, prad_indices=None)
		if cancer_type == 1:
			features = prep.select_features(n, featureselect, brca_indices=None, prad_indices=train)

		train_expression_data = prep.dat[train_expression][:, features]
		train_event_data = prep.dat[train_event]
		test_expression_data = prep.dat[test_expression][:, features]
		test_event_data = prep.dat[test_event]
		probas_ = model[i].predict_proba(test_expression_data)
		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(test_event_data, probas_[:, 1])
		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

	mean_tpr /= len(folds)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, 'k--',
	         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(name)
	plt.legend(loc="lower right")
	plt.savefig('../images/' + name)
	plt.clf()


def PCA_plot(data=None, model_pickle=None, model=None, cancer_type=0, pltname='PC1PC2'):
	if model_pickle is not None:
		model = pickle.load(open(model_pickle, 'rb'))

	train_expression = cancer_type * 4
	test_expression = cancer_type * 4 + 1
	train_event = cancer_type * 4 + 2
	test_event = cancer_type * 4 + 3

	prediction = model.predict(data[test_expression])

	tp = np.logical_and(prediction == 1, data[test_event] == 1)  # true positives
	fp = np.logical_and(prediction == 1, data[test_event] == 0)  # false positives
	tn = np.logical_and(prediction == 0, data[test_event] == 0)  # true positive
	fn = np.logical_and(prediction == 0, data[test_event] == 1)  # false negative

	pca = PCA(2)
	projected_data = pca.fit(data[test_expression]).transform(data[test_expression])

	#fig, ax = plt.subplots(ncols=1)

	colors = iter(cm.rainbow(np.linspace(0,1,5)));
	plt.scatter(projected_data[fp,0],projected_data[fp,1], c='b', s=100, label='False Pos')
	plt.scatter(projected_data[tn,0],projected_data[tn,1], c='k', s=100, label='True Neg')
	plt.scatter(projected_data[fn,0],projected_data[fn,1], c='r', s=100, label='False Neg')
	plt.scatter(projected_data[tp,0],projected_data[tp,1], c='g', s=300, label='True Pos')


	plt.xlabel("PC-1 of Test Data")
	plt.ylabel("PC-2 of Test Data")
	plt.title(pltname)
	plt.legend(loc='center left', bbox_to_anchor=(0.85,0.5))


	plt.show()