import pickle
import numpy as np
import scipy as sc 
from scipy import stats
from scipy.stats import norm
import matplotlib.mlab as mlab
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


brcadatapath = '../data/BRCAdataset.p'
brcafollowuppath = '../data/BRCAfollowUpDataset.p'
praddatapath = '../data/PRADdataset.p'
pradfollowuppath = '../data/PRADfollowUpDataset.p'

brca_expression = pickle.load(open(brcadatapath, 'rb'))
prad_expression = pickle.load(open(praddatapath, 'rb'))

brca_expression = (brca_expression[1::,1::]).astype(float)  # drop header row
prad_expression = (prad_expression[1::,1::]).astype(float)  # drop header row

brca_histdat = brca_expression[0:359]
prad_histdat = prad_expression[0:279]

print "Making Raw BRCA Expression Histogram"
n, bins, patches = plt.hist(brca_histdat.transpose(), bins=range(0,4500,50))
(mu, sigma) = norm.fit(brca_histdat)
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
plt.title(r'$\mathrm{Histogram\ of\ Raw\ BRCA\ Expression:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
plt.show()

print "Making Raw PRAD Expression Histogram"
n, bins, patches = plt.hist(prad_histdat.transpose(), bins=range(0,4500,50))
(mu, sigma) = norm.fit(prad_histdat)
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
plt.title(r'$\mathrm{Histogram\ of\ Raw\ PRAD\ Expression:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
plt.show()

#log transform data
brca_expression = np.log(brca_expression + 1)
prad_expression = np.log(prad_expression + 1)
    
# load tumor event data
brca_event_data = pickle.load(open(brcafollowuppath, 'rb'))
prad_event_data = pickle.load(open(pradfollowuppath, 'rb'))

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

brca_expression_train = scale(brca_expression_train)
brca_expression_test = scale(brca_expression_test)
prad_expression_train = scale(prad_expression_train)
prad_expression_test = scale(prad_expression_test)
        
brca_histdat = brca_expression_train
prad_histdat = prad_expression_train

print "Making Norm BRCA Expression Histogram"
n, bins, patches = plt.hist(brca_histdat.transpose(), bins=np.arange(-10,10,.25))
(mu, sigma) = norm.fit(brca_histdat)
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
plt.title(r'$\mathrm{Histogram\ of\ Processed\ BRCA\ Expression:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
plt.show()

print "Making Norm PRAD Expression Histogram"
n, bins, patches = plt.hist(prad_histdat.transpose(), bins=np.arange(-10,10,.25))
(mu, sigma) = norm.fit(prad_histdat)
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
plt.title(r'$\mathrm{Histogram\ of\ Processed\ PRAD\ Expression:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
plt.show()
