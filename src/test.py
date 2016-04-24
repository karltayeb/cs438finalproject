from prep import *
from sklearn.svm import SVC
from sklearn import cross_validation
import pickle

#data = Prep(50)   #prep data, take top 50 features
#brca_svm = SVC(kernel='linear', C=1, probability=True).fit(data.brca_expression_train, data.brca_event_train)
#pickle.dump(brca_svm, open('brca50_svm.p','wb'))
#prad_svm = SVC(kernel='linear', C=1, probability=True).fit(data.prad_expression_train, data.prad_event_train)
#pickle.dump(brca_svm, open('prad50_svm.p','wb'))

data = Prep(100)
brca_svm = SVC(kernel='linear', C=1, probability=True).fit(data.brca_expression_train, data.brca_event_train)
pickle.dump(brca_svm, open('brca100_svm.p', 'wb'))
prad_svm = SVC(kernel='linear', C=1, probability=True).fit(data.prad_expression_train, data.prad_event_train)
pickle.dump(brca_svm, open('prad100_svm.p','wb'))

data = Prep(150)
brca_svm = SVC(kernel='linear', C=1, probability=True).fit(data.brca_expression_train, data.brca_event_train)
pickle.dump(brca_svm, open('brca150_svm.p', 'wb'))
prad_svm = SVC(kernel='linear', C=1, probability=True).fit(data.prad_expression_train, data.prad_event_train)
pickle.dump(brca_svm, open('prad150_svm.p','wb'))

#brca_scores = cross_validation.cross_val_score(
#		brca_svm, data.brca_expression_test, data.brca_event_test, cv=5, scoring='f1_weighted')