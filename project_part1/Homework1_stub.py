#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import sklearn
import numpy
import random
import gzip
import math


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[4]:


f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

len(dataset)
# In[5]:


type(dataset)


# In[6]:


answers = {} # Put your answers to each question in this dictionary`


# In[7]:


dataset[0]


# In[8]:


### Question 1


# In[9]:


numExclamation = [d['review_text'].count('!') for d in dataset]
rating = [d['rating'] for d in dataset]


# In[10]:


plt.scatter(numExclamation, rating, color='grey')
plt.xlim(0, 100)
plt.ylim(0.5, 5.5)
plt.xlabel("Number of Excalations")
plt.ylabel("Rating")
plt.title("Rating vs. Number of Excalamations")
plt.show()


# In[11]:


X = numpy.matrix([[1,l] for l in numExclamation]) # Note the inclusion of the constant term
y = numpy.matrix(rating).T


# In[12]:


model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)


# In[13]:


#theta = model.coef_
#theta


# In[14]:


theta,residuals,rank,s = numpy.linalg.lstsq(X, y, rcond=None)
theta


# In[15]:


numpy.linalg.inv(X.T*X)*X.T*y


# In[16]:


xplot = numpy.arange(0,5501,10)
yplot = [(theta[0] + theta[1]*x).item() for x in xplot]


# In[17]:


plt.scatter(numExclamation, rating, color='grey')
plt.plot(numpy.array(xplot), yplot, color = 'k', linestyle = '--',         label = r"$3.983 + 1.193 \times 10^{-4} \mathit{length}$")
plt.xlim(0, 55)
plt.ylim(0.5, 5.5)
plt.xlabel("Number of Excalamations")
plt.ylabel("Rating")
plt.title("Rating vs. Number of Excalamations")
plt.legend(loc='lower right')
plt.show()


# In[18]:


y_pred = model.predict(X)


# In[19]:


sse = sum([x**2 for x in (y - y_pred)])


# In[20]:


mse = float(sse / len(y))
mse


# In[21]:


theta0 = float(theta[0])
theta1 = float(theta[1])
answers['Q1'] = [theta0,theta1,mse] # when in doubt, restart kernel.
answers


# In[22]:


assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)


# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


### Question 2


# In[24]:


def feature(datum): 
# appends onto rows/col of dataset onto new list # inputting what is acting as predictor
    feat = [1]
    feat.append(len(datum['review_text'])) # append length of review_text
    feat.append(datum['review_text'].count('!')) # append number of exclamations onto dataset
    return feat


# In[25]:


X = numpy.matrix([feature(d) for d in dataset])
#X = numpy.matrix([[1,l] for l in numExclamation]) # Note the inclusion of the constant term
#y = numpy.matrix(rating).T


# In[26]:


modelQ2 = sklearn.linear_model.LinearRegression(fit_intercept=False)
modelQ2.fit(X, y)


# In[27]:


thetaQ2 = modelQ2.coef_
thetaQ2


# In[28]:


y_pred = modelQ2.predict(X) 
sse = sum([r**2 for r in y - y_pred])
mse = sse / len(y) 


# In[29]:


theta0 = float(thetaQ2[0][0])
theta1 = float(thetaQ2[0][1])
theta2 = float(thetaQ2[0][2])
mse = float(mse)
answers['Q2'] = [theta0, theta1, theta2, mse] 
answers['Q2']


# In[30]:


assertFloatList(answers['Q2'], 4)


# In[ ]:





# In[ ]:





# In[ ]:





# In[31]:


### Question 3


# In[32]:


def feature_Q3(datum, deg):                               # building ontop of our base model
    # feature for a specific polynomial degree
    feat = [1]                                         # add in theta_3
    for i in range (1,deg+1):
        num_ofExc = datum['review_text'].count('!')    
        feat.append(num_ofExc**i)                      # add in [number of !]
    return feat                                        

# with new polynomial function, we have to remove [length] from q2 model, and 

    
    


# In[33]:


mses = []

 
for i in range(1,6): # up to pow 5
    X = numpy.matrix([feature_Q3(d,i) for d in dataset]) # this tacks on feat() from the base model in q1
    model = sklearn.linear_model.LinearRegression(fit_intercept=False) # modeling our data, and fitting it
    model.fit(X, y) # provides new theta for each argument term within model: feat[]
    
    theta = model.coef_  # calculating theta 
    y_pred = model.predict(X)               # prediction model, predicting future values to calculate mse.
    
    sse = sum([r**2 for r in y - y_pred])
    mse = float(sse / len(y))              # MSE = SSE / DFE  # DFE = N-k = # of obs - # of groups   # DFE is Deg of freedom
    mses.append(mse)
    #mses.append(float(sse / (len(y)-4))) # k=4 groups, N=len(y)=10000 observations 
    
mses


# In[34]:


#theta = model.coef_  # calculating theta
#theta 


# In[35]:


#y_pred = model.predict(X)               # prediction model, predicting future values to calculate mse.
#sse = sum([r**2 for r in y - y_pred])
#mses = []           # MSE = SSE / DFE  # DFE = N-k = # of obs - # of groups   # DFE is Deg of freedom
#mses.append(float(sse / (len(y)-4))) # k=4 groups, N=len(y)=10000 observations 


# In[36]:


answers['Q3'] = mses   # all of the mse's  # but isn't MSE used on a group of data? (ref. math183)
mses


# In[37]:


assertFloatList(answers['Q3'], 5) # List of length 5


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[38]:


### Question 4


# In[39]:


#X = [b[] for b in data]
#y = [""]
X = [d['review_text'] for d in dataset]  # will use feature() here, so must start without [number of !]
Y = [d['rating'] for d in dataset]       # trying to predict ratings, so it would be the dependent variables


# In[40]:


# Splitting data into training and test set - ref. week 2 lec notes.
Training_DS = dataset[:len(dataset)//2]  # full training dataset
# X_train = X[:len(X)//2] # rating
y_train = Y[:len(Y)//2] # excalamation  # trying to predict ratings, so it would be the dependent variables


Test_DS = dataset[len(dataset)//2:]      # full test dataset
# X_test = X[len(X)//2:] # rating          # equiv to dataset / 2 ?
y_test = Y[len(Y)//2:] # excalamation


# In[41]:


# Transpose matrices, so we can perform operations
y_train = numpy.matrix(y_train).T
y_test = numpy.matrix(y_test).T


# In[42]:


#mod = linear_model.LogisticRegression(C=1.0)
#mod.fit(X_train, y_train)   


# In[43]:


#train_predictions = mod.predict(X_train)
#test_predictions = mod.predict(X_test)   # equivalent to y_pred from prev. questions # we need this one.


# In[44]:


mses = [] # similar to question 3 but w/ test data

for i in range(1,6): # up to pow 5
    X_test = numpy.matrix([feature_Q3(d,i) for d in Test_DS])                   # this tacks on feat() from the base model in q1
    model_Q4 = sklearn.linear_model.LinearRegression(fit_intercept=False) # modeling our data, and fitting it
    model_Q4.fit(X_test, y_test) # provides new theta for each argument term within model: feat[]
    
    thetaQ4 = model_Q4.coef_  # calculating theta 
    y_pred = model_Q4.predict(X_test)           # prediction model, predicting future values to calculate mse.
    
    sse = sum([r**2 for r in y_test - y_pred])  # do we use y_train or y_test? prob test.
    mse = float(sse / len(y_test))              # MSE = SSE / DFE  # DFE = N-k = # of obs - # of groups   # DFE is Deg of freedom
    mses.append(mse)
    #mses.append(float(sse / (len(y)-4))) # k=4 groups, N=len(y)=10000 observations
    
mses


# In[ ]:





# In[45]:


# calculate MSE on test set?
#sse = sum([r**2 for r in y_test - test_predictions]) # do we use y_train or y_test? prob test.
#sse #LinAlgError msg : imbalance between X and y matrix


# In[46]:


answers['Q4'] = mses


# In[47]:


assertFloatList(answers['Q4'], 5)


# In[48]:


answers['Q4']


# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


### Question 5


# In[ ]:





# In[50]:


from sklearn.metrics import mean_absolute_error 

MAEs = [] # similar to question 3 but w/ test data

for i in range(1,6): # up to pow 5
    X_test = numpy.matrix([feature_Q3(d,i) for d in Test_DS])                # this tacks on feat() from the base model in q1
    model_Q5 = sklearn.linear_model.LinearRegression(fit_intercept=False)         # modeling our data, and fitting it
    model_Q5.fit(X_test, y_test)                    # provides new theta for each argument term within model: feat[]
    thetaQ5 = model_Q5.coef_                          # calculating theta 
    y_pred = model_Q5.predict(X_test)                 # prediction model, predicting future values to calculate mse.
    
    MAE_instance = mean_absolute_error(y_test,y_pred) # temporary container for each MAE :: feat.
    MAEs.append(MAE_instance)
    # MAE = numpy.append(MAE,MAE_instance)

MAEs
    
    
    #sse = sum([r**2 for r in y_test - y_pred]) # do we use y_train or y_test? prob test.
    #mse = float(sse / len(y))              # MSE = SSE / DFE  # DFE = N-k = # of obs - # of groups   # DFE is Deg of freedom
    #mses.append(mse)
    #mses.append(float(sse / (len(y)-4))) # k=4 groups, N=len(y)=10000 observations
    


# In[51]:


# local minimum which should be the trivial predictor # minimum value which makes loss function min.
MAE = []
MAE = min(MAEs)


# In[ ]:





# In[52]:


answers['Q5'] = MAE


# In[53]:


assertFloat(answers['Q5'])


# In[54]:


answers['Q5']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[55]:


### Question 6


# In[56]:


f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))


# In[57]:


len(dataset)


# In[58]:


y = [d['user/gender'] == ('Female') for d in dataset] # theta0
#X = numpy.matrix([d['review/text'].count('!') for d in dataset]) # theta1
X = [[1, d['review/text'].count('!')] for d in dataset]


# In[59]:


mod = sklearn.linear_model.LogisticRegression()
mod.fit(X,y)


# In[60]:


predictions = mod.predict(X) # Binary vector of predictions
correct = predictions == y # Binary vector indicating which predictions were correct
sum(correct) / len(correct)


# In[61]:


# True positive, false positive, true negative, false negative
TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])


# In[62]:


print("TP = " + str(TP))
print("FP = " + str(FP))
print("TN = " + str(TN))
print("FN = " + str(FN))

# How can TP & FP be 0? using balanced error rate fixes it.
# female data was weighted more?


# In[63]:


(TP + TN) / (TP + FP + TN + FN)


# In[64]:


# True positive rate, true negative rate, Balanced error rate
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
TPR,TNR
BER = 1 - 1/2 * (TPR + TNR)
BER


# In[ ]:





# In[ ]:





# In[65]:


answers['Q6'] = [TP, TN, FP, FN, BER]


# In[66]:


assertFloatList(answers['Q6'], 5)


# In[67]:


answers['Q6']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[68]:


### Question 7


# In[69]:


y = [d['user/gender'] == ('Female') for d in dataset]                
# X = numpy.matrix([d['review/text'].count('!') for d in dataset])   
X = [[1, d['review/text'].count('!')] for d in dataset]   


# In[70]:


mod = sklearn.linear_model.LogisticRegression(class_weight='balanced')  # create logistic regression
mod.fit(X,y)                          # fit model
predictions = mod.predict(X)          # create model with prediction


# In[71]:


correct = predictions == y            # Binary vector indicating which predictions were correct


# In[72]:


# calculate True positive, False positive, True negative, False negative
TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])

# True positive rate, True negative Rate, Balanced Error rate
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
TPR,TNR
BER = 1 - 1/2 * (TPR + TNR)
BER


# In[ ]:





# In[73]:


answers["Q7"] = [TP, TN, FP, FN, BER]


# In[74]:


assertFloatList(answers['Q7'], 5)


# In[75]:


answers["Q7"]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[76]:


### Question 8


# In[77]:


K = [1, 10, 100, 1000, 10000]  # list of precision
precisionList = []             # empty precision list


# In[78]:


scores = mod.decision_function(X)               # score labels
scoreslabels = list(zip(scores, y))
scoreslabels.sort(reverse=True)
sortedlabels = [x[1] for x in scoreslabels]    # list of sorted labels


# In[79]:


# precision
retrieved = sum(predictions)
relevant = sum(y)
intersection = sum([y and p for y,p in zip(y,predictions)])


# In[80]:


intersection / retrieved
# recall
intersection / relevant


# In[ ]:





# In[81]:


for toPrecision in K:
    # sum(sortedlabels[:10]) / 10
    prec_instance = sum(sortedlabels[:toPrecision]) / toPrecision # bring data up to precision
    precisionList.append(prec_instance)                           # append data to precision
precisionList
    #precision = TP / (TP + FP) # precision formula?
 


# In[ ]:





# In[ ]:





# In[ ]:





# In[82]:


answers['Q8'] = precisionList


# In[83]:


assertFloatList(answers['Q8'], 5) #List of five floats


# In[84]:


f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




