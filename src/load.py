import pickle
import numpy as np
import scipy as sc 	

brcadatapath = '../data/BRCAdataset.p'
brcafollowuppath = '../data/BRCAfollowUpDataset.p'
praddatapath = '../data/PRADdataset.p'
pradfollowuppath = '../data/PRADfollowUpDataset.p'

def loadData():
	# load expr
	brca_expression = pickle.load(open(brcadatapath, 'rb'))
	prad_expression = pickle.load(open(praddatapath, 'rb'))

	brca_expression = (brca_expression[1::,1::]).astype(float)  # drop header row
	prad_expression = (prad_expression[1::,1::]).astype(float)  # drop header row

	# load tumor event data
	brca_event_data = pickle.load(open(brcafollowuppath, 'rb'))
	prad_event_data = pickle.load(open(pradfollowuppath, 'rb'))

	# append True/False corresponding to Yes/No
	brca_event = brca_event_data[:,1] == 'YES'
	prad_event = prad_event_data[:,1] == 'YES'

	# seperate into training and test sets, for now randomly pick 70% of the data

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
