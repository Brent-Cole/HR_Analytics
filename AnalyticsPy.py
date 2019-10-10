# Load libraries
import pandas
import numpy as np
from xlwt import Workbook
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

np.set_printoptions(threshold=np.inf)

# Load datasets

df = pandas.read_csv("Training.xls", sep="\t", header=0)
testData = pandas.read_excel("testing.xls")

# descriptions on the data sets
print(testData.describe())
#print(df.describe())


# box and whisker plots of the training data
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)

plt.show()

# histograms of the training data
df.hist()
plt.show()


# scatter plot matrix   Commented out because it takes too long to load
#scatter_matrix(df)
#plt.show()

#creating a subset of the data
tf = df[df.columns[6:14]]

# Split-out validation of tf
array = tf.values
X = array[:,0:6]
Y = array[:,7]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'f1'


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('XGB', XGBClassifier()))


# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Creating new data set for the new tests with the whole test data
tf = testData[testData.columns[6:13]]
array = tf.values
X = array[:,0:6]


# Testing all of the different models listed above

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

nb = GaussianNB()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


#based on the training data  I chose the Gradient boost for the main tuning
xb = XGBClassifier(learning_rate=0.1, n_estimators=200, max_depth=4, min_child_weight=7, gamma=.4, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=3,seed=29)
xb.fit(X_train, Y_train)
predictions = xb.predict(X)

#Save the predictions in the workbook
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')
temp = np
temp = predictions

#I needed to format it for submission with one number per row
for x in range(len(temp)):
    sheet1.write(x,0, str(temp[x]))

wb.save('predict.xls')

