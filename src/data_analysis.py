from prep import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def shapiro_on_transformations():
	data = Prep(-1, logtransform=True, scaledata=True)
	lmda = np.arange(0, 0.5, 0.05)

	print 'Shapiro Wilk Test: mean, median of features p < 0.0001'

	print 'BRCA training data'
	shapiro = np.apply_along_axis(stats.shapiro, 0, data.brca_expression_train)
	print 'n/a', ':', np.mean(shapiro[0,np.where(shapiro[1,:] < 0.0001)[0]]), np.median(shapiro[0,np.where(shapiro[1,:] < 0.001)[0]])
	for l in lmda:
		brca_boxcox = np.apply_along_axis(stats.boxcox, 0, (data.brca_expression_train + 1), l)
		shapiro = np.apply_along_axis(stats.shapiro, 0, brca_boxcox)
		print l, ':', np.mean(shapiro[0,np.where(shapiro[1,:] < 0.0001)[0]]), np.median(shapiro[0,np.where(shapiro[1,:] < 0.001)[0]])

	print 'PRAD training data'
	shapiro = np.apply_along_axis(stats.shapiro, 0, data.brca_expression_train)
	print 'n/a', ':', np.mean(shapiro[0,np.where(shapiro[1,:] < 0.0001)[0]]), np.median(shapiro[0,np.where(shapiro[1,:] < 0.001)[0]])
	for l in lmda:
		prad_boxcox = np.apply_along_axis(stats.boxcox, 0, (data.prad_expression_train + 1), l)
		shapiro = np.apply_along_axis(stats.shapiro, 0, prad_boxcox)
		print l, ':', np.mean(shapiro[0,np.where(shapiro[1,:] < 0.0001)[0]]), np.median(shapiro[0,np.where(shapiro[1,:] < 0.001)[0]])


def logtransform_vs_not_QQ():
	data = Prep(1, logtransform=False, scaledata=True)
	logdata = Prep(1, logtransform=True, scaledata=True)

	n = stats.norm()
	m = data.brca_expression_train.size
	obs = np.sort((data.brca_expression_train))
	logobs = np.sort((logdata.brca_expression_train))
	th = np.arange(1/float(m),1+(1/float(m)),1/float(m))

	fig, ax = plt.subplots(ncols=2)
	ax[0].scatter(th,obs)
	x = np.linspace(*ax[0].get_xlim())
	ax[0].plot(x, x)

	ax[1].scatter(th,logobs)
	x = np.linspace(*ax[1].get_xlim())
	ax[1].plot(x, x)

	plt.show()
	return data, logdata

"""

fig, ax = plt.subplots(ncols=1)

# since data is scaled
n = data.brca_expression_train.shape[0]

for i in range(2, 10):
	print i, stats.shapiro(data.brca_expression_train[:,i])
	obs = np.sort(data.brca_expression_train[:,i])
	th = np.arange(1/float(n),1+(1/float(n)),1/float(n))
	fig, ax = plt.subplots(ncols=1)
	ax.scatter(th,obs)
	x = np.linspace(*ax.get_xlim())
	ax.plot(x, x)
	ax.set_title(i)


plt.show()
