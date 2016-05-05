from prep import *
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn import cross_validation
import pickle


data = Prep(150, logtransform=True, scaledata=True)

brca_svm = SVC(kernel='linear', C=1, probability=True).fit(data.brca_expression_train, data.brca_event_train)
prad_svm = SVC(kernel='linear', C=1, probability=True).fit(data.prad_expression_train, data.prad_event_train)