import pickle
import numpy as np
import scipy as sc 
from scipy import stats
from sklearn.preprocessing import scale

class Prep:

	brcadatapath = '../data/BRCAdataset.p'
	brcafollowuppath = '../data/BRCAfollowUpDataset.p'
	praddatapath = '../data/PRADdataset.p'
	pradfollowuppath = '../data/PRADfollowUpDataset.p'

	def __init__ (self, n=-1, logtransform=False, scaledata=False, featureselect='mixed'):
		self.brca_expression_train, self.brca_expression_test, self.brca_event_train, self.brca_event_test, self.prad_expression_train, self.prad_expression_test, self.prad_event_train, self.prad_event_test = self.loadData(logtransform, scaledata)
		self.select_features(n, featureselect)
		self.dat = (self.brca_expression_train, self.brca_expression_test, self.brca_event_train, self.brca_event_test, self.prad_expression_train, self.prad_expression_test, self.prad_event_train, self.prad_event_test)

	def loadData(self, logtransform, scaledata):
		# load expr
		brca_expression = pickle.load(open(self.brcadatapath, 'rb'))
		prad_expression = pickle.load(open(self.praddatapath, 'rb'))

		brca_expression = (brca_expression[1::,1::]).astype(float)  # drop header row
		prad_expression = (prad_expression[1::,1::]).astype(float)  # drop header row

		#log transform data
		if logtransform:
			brca_expression = np.log(brca_expression + 1)
			prad_expression = np.log(prad_expression + 1)

		# load tumor event data
		brca_event_data = pickle.load(open(self.brcafollowuppath, 'rb'))
		prad_event_data = pickle.load(open(self.pradfollowuppath, 'rb'))

		# append True/False corresponding to Yes/No
		brca_event = brca_event_data[:,1] == 'YES'
		prad_event = prad_event_data[:,1] == 'YES'

		# seperate into training and test sets, for now pick front 60% of the data
		# as training set
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


		# scale the expression data to have mean 0 variance 1
		if scaledata:
			brca_expression_train = scale(brca_expression_train)
			brca_expression_test = scale(brca_expression_test)
			prad_expression_train = scale(prad_expression_train)
			prad_expression_test = scale(prad_expression_test)

		return brca_expression_train, brca_expression_test, brca_event_train, brca_event_test, prad_expression_train, prad_expression_test, prad_event_train, prad_event_test

	def select_features(self, n, featureselect):
		"""
		Rank features by signal-to-noise
		Takes n from brca, n from prad, and n overlap
		this is an attempt to pick meaningful features for both prediction tasks
		"""

		brca_new_tumor = np.where(self.brca_event_train == True)[0]
		brca_no_tumor = np.where(self.brca_event_train == False)[0]
		prad_new_tumor = np.where(self.prad_event_train == True)[0]
		prad_no_tumor = np.where(self.prad_event_train == False)[0]

		#brca_p = stats.ttest_ind(self.brca_expression_train[brca_new_tumor,:], self.brca_expression_train[brca_no_tumor,:], axis=0)[1]
		#prad_p = stats.ttest_ind(self.prad_expression_train[prad_new_tumor,:], self.brca_expression_train[prad_no_tumor,:], axis=0)[1]

		brca_snr = np.abs(
					np.mean(self.brca_expression_train[brca_new_tumor,:], axis=0)
					- np.mean(self.brca_expression_train[brca_no_tumor,:], axis=0)
					) / 2
		prad_snr = np.abs(
					np.mean(self.prad_expression_train[prad_new_tumor,:], axis=0)
					- np.mean(self.brca_expression_train[prad_no_tumor, :], axis=0)
					) / 2

		#high_p = np.amax(np.c_[brca_p, prad_p], axis=1)
		low_snr = np.amin(np.c_[brca_snr, prad_snr], axis=1)
		#high_snr = np.amax(np.c_[brca_snr, prad_snr], axis=1)

		#featurerank = np.argsort(high_p)
		brca_featurerank = np.argsort(-1 * brca_snr)
		prad_featurerank = np.argsort(-1 * prad_snr)
		joint_featurerank = np.argsort(-1 * low_snr)

		if featureselect is 'mixed':
			if (n != -1):
				indices = np.empty(0)
				i = 0
				while indices.size < n:
					if not (indices == np.where(brca_featurerank == i)[0]).any() and indices.size < n:
						indices = np.append(indices, np.where(brca_featurerank == i)[0])
					if not (indices == np.where(prad_featurerank == i)[0]).any() and indices.size < n:
						indices = np.append(indices, np.where(prad_featurerank == i)[0])
					if not (indices == np.where(joint_featurerank == i)[0]).any() and indices.size < n:
						indices = np.append(indices, np.where(joint_featurerank == i)[0])
						i += 1

				indices = indices.astype(int)
				self.brca_expression_train = self.brca_expression_train[:,indices]
				self.brca_expression_test = self.brca_expression_test[:,indices]
				self.prad_expression_train = self.prad_expression_train[:,indices]
				self.prad_expression_test = self.prad_expression_test[:,indices]

		if featureselect is 'seperate':
			if (n != -1):
				brca_indices = np.where(brca_featurerank < n)[0].astype(int)
				prad_indices = np.where(prad_featurerank < n)[0].astype(int)

				self.brca_expression_train = self.brca_expression_train[:,brca_indices]
				self.brca_expression_test = self.brca_expression_test[:,brca_indices]
				self.prad_expression_train = self.prad_expression_train[:,prad_indices]
				self.prad_expression_test = self.prad_expression_test[:,prad_indices]

		if featureselect is 'overlap':
			if (n != -1):
				indices = np.where(joint_featurerank < n)[0].astype(int)

				self.brca_expression_train = self.brca_expression_train[:,indices]
				self.brca_expression_test = self.brca_expression_test[:,indices]
				self.prad_expression_train = self.prad_expression_train[:,indices]
				self.prad_expression_test = self.prad_expression_test[:,indices]				
