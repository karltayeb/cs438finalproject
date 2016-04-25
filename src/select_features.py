from load import *
import numpy as np
from scipy import stats

# load data
def select_features():
	brca_expression_train, brca_expression_test, brca_event_train, brca_event_test, prad_expression_train, prad_expression_test, prad_event_train, prad_event_test = loadData()

	brca_new_tumor = np.where(brca_event_train == True)[0]
	brca_no_tumor = np.where(brca_event_train == False)[0]
	prad_new_tumor = np.where(prad_event_train == True)[0]
	prad_no_tumor = np.where(prad_event_train == False)[0]

	brca_p = stats.ttest_ind(brca_expression_train[brca_new_tumor,:], brca_expression_train[brca_no_tumor,:], axis=0)[1]
	prad_p = stats.ttest_ind(prad_expression_train[prad_new_tumor,:], brca_expression_train[prad_no_tumor,:], axis=0)[1]

	high_p = np.amax(np.c_[brca_p, prad_p], axis=1)

	featurerank = np.argsort(high_p)

	indices = np.where(featurerank < n)