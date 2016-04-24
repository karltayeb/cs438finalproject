import pickle
import numpy as np
import scipy as sc 
from scipy import stats

class Prep:

	brcadatapath = '../data/BRCAdataset.p'
	brcafollowuppath = '../data/BRCAfollowUpDataset.p'
	praddatapath = '../data/PRADdataset.p'
	pradfollowuppath = '../data/PRADfollowUpDataset.p'

	def __init__ (self, n):
		self.brca_expression_train, self.brca_expression_test, self.brca_event_train, self.brca_event_test, self.prad_expression_train, self.prad_expression_test, self.prad_event_train, self.prad_event_test = self.loadData()
		self.select_features(n)

	def loadData(self):
		# load expr
		brca_expression = pickle.load(open(self.brcadatapath, 'rb'))
		prad_expression = pickle.load(open(self.praddatapath, 'rb'))

		brca_expression = (brca_expression[1::,1::]).astype(float)  # drop header row
		prad_expression = (prad_expression[1::,1::]).astype(float)  # drop header row

		# load tumor event data
		brca_event_data = pickle.load(open(self.brcafollowuppath, 'rb'))
		prad_event_data = pickle.load(open(self.pradfollowuppath, 'rb'))

		# append True/False corresponding to Yes/No
		brca_event = brca_event_data[:,1] == 'YES'
		prad_event = prad_event_data[:,1] == 'YES'

		# seperate into training and test sets, for now pick front 70% of the data

		brca_div = int(brca_event.size * 0.7)
		# divide into train and test
		brca_expression_train = brca_expression[0:brca_div, :]
		brca_expression_test = brca_expression[brca_div::, :]
		brca_event_train = brca_event[0:brca_div]
		brca_event_test = brca_event[brca_div::]

		prad_div = int(prad_event.size * 0.7)
		# divide into train and test
		prad_expression_train = prad_expression[0:prad_div, :]
		prad_expression_test = prad_expression[prad_div::, :]
		prad_event_train = prad_event[0:prad_div]
		prad_event_test = prad_event[prad_div::]

		return brca_expression_train, brca_expression_test, brca_event_train, brca_event_test, prad_expression_train, prad_expression_test, prad_event_train, prad_event_test

	def select_features(self, n):
		"""
		Does significance testing between tumor and non-new-tumor groups in both cancers
		takes the overlap in significant features
		ranks figures by the most significant in both
		truncates data to featurespace we are considering (top n feature)
		"""

		brca_new_tumor = np.where(self.brca_event_train == True)[0]
		brca_no_tumor = np.where(self.brca_event_train == False)[0]
		prad_new_tumor = np.where(self.prad_event_train == True)[0]
		prad_no_tumor = np.where(self.prad_event_train == False)[0]

		brca_p = stats.ttest_ind(self.brca_expression_train[brca_new_tumor,:], self.brca_expression_train[brca_no_tumor,:], axis=0)[1]
		prad_p = stats.ttest_ind(self.prad_expression_train[prad_new_tumor,:], self.brca_expression_train[prad_no_tumor,:], axis=0)[1]

		high_p = np.amax(np.c_[brca_p, prad_p], axis=1)

		featurerank = np.argsort(high_p)

		indices = np.where(featurerank < n)[0]

		self.brca_expression_train = self.brca_expression_train[:,indices]
		self.brca_expression_test = self.brca_expression_test[:,indices]
		self.prad_expression_train = self.prad_expression_train[:,indices]
		self.prad_expression_test = self.prad_expression_test[:,indices]
