#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[4]:


f = open("5year.arff", 'r')


# In[5]:


# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)


# In[6]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[7]:


answers = {} # Your answers


# In[8]:


def accuracy(predictions, y):
    correct = predictions == y
    return sum(correct) / len(y)     # accuracy formula


# In[9]:


def BER(predictions, y): # BER calculations - chapter 3 wkbk
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    BER = 1 - (1/2) * (TPR + TNR)
    
    return BER


# In[10]:


### Question 1


# In[11]:


mod = linear_model.LogisticRegression(C=1) # trains logistic regressor
mod.fit(X,y)

pred = mod.predict(X)


# In[12]:


acc1 = accuracy(pred,y)
ber1 = BER(pred,y)


# In[13]:


answers['Q1'] = [acc1, ber1] # Accuracy and balanced error rate
answers['Q1'] # why is accuracy so low (3%)? BER looks okay.


# In[14]:


assertFloatList(answers['Q1'], 2)


# In[15]:


### Question 2


# In[16]:


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)


# In[17]:


acc2 = accuracy(pred,y)
ber2 = BER(pred,y)


# In[18]:


answers['Q2'] = [acc2, ber2]
answers['Q2'] # now ber and acc are literally 'balanced'?


# In[19]:


assertFloatList(answers['Q2'], 2)


# In[20]:


### Question 3


# In[21]:


random.seed(3)
random.shuffle(dataset)


# In[22]:


X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]


# In[23]:


# splits data into training (50%), validation (25%), test (25%)
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[24]:


len(Xtrain), len(Xvalid), len(Xtest)


# In[25]:


# training on the training set - for train
mod_train = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod_train.fit(Xtrain,ytrain)

pred_train = mod_train.predict(Xtrain)

# calculate BER
berTrain = BER(pred_train,ytrain)


# In[26]:


# training on the training set - for validation
mod_valid = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod_valid.fit(Xtrain,ytrain) # (Xvalid,yvalid)

pred_valid = mod_valid.predict(Xvalid)

# calculate BER
berValid = BER(pred_valid,yvalid)


# In[27]:


# training on the training set - for test
mod_test = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod_test.fit(Xtrain,ytrain) # (Xtest,ytest)

pred_test = mod_test.predict(Xtest)

# calculate BER
berTest = BER(pred_test,ytest)


# In[28]:


# what does "training on the training set" mean exactly? (in code)


# In[29]:


answers['Q3'] = [berTrain, berValid, berTest]
answers['Q3']


# In[30]:


assertFloatList(answers['Q3'], 3)


# In[31]:


### Question 4


# In[32]:


berList = []


# In[33]:


# for validation BER - regularization pipeline
#C = [10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

def regPiped(C, Xvalid, yvalid):
    outList = []
    # each value of C has a corresponding BER calculation.
    for c_values in C:

        mod_vPiped = linear_model.LogisticRegression(C=c_values, class_weight='balanced') # don't forget C=...
        mod_vPiped.fit(Xvalid,yvalid) # curr problem: unable to 
        
        pred_vPiped = mod_vPiped.predict(Xvalid)
        
        iBER = BER(pred_vPiped,yvalid) # current loop's BER value
        
        outList.append(iBER) # add BER value to list of BERs
    
    return outList

berList = regPiped(C, Xvalid, yvalid)


# In[34]:


answers['Q4'] = berList
answers['Q4']


# In[35]:


assertFloatList(answers['Q4'], 9)


# In[36]:


### Question 5


# In[37]:


# ideally we want BER of 0 for a perfect classifier. ref. classification lec slide.
# and 0.5 for a random/naive classifier.
min(berList) # the smallest value in berList was 0.3131420817987982
# Correspondingly, shows that C=100 performed the best.
bestC = 100 


# In[38]:


# now, calculate BER on the test set w/ this C-value
# same calculations as prev question

mod_test = linear_model.LogisticRegression(C=100, class_weight='balanced') # using C=100
mod_test.fit(Xtest,ytest)

pred_test = mod_test.predict(Xtest)

# calculate BER
ber5 = BER(pred_test,ytest) # previously at ~0.19, now at ~0.18. Which is an improvement.


# In[39]:


answers['Q5'] = [bestC, ber5]
answers['Q5']


# In[40]:


assertFloatList(answers['Q5'], 2)


# In[41]:


### Question 6


# In[42]:


f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))


# In[43]:


dataTrain = dataset[:9000]
dataTest = dataset[9000:]


# In[44]:


dataset[0]


# In[ ]:





# In[ ]:





# In[45]:


# Some data structures you might want

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataset: #dataTrain
    user,item = d['user_id'], d['book_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    ratingDict[(user,item)] = d['rating']
    # reviewsPerItem = 
    # reviewsPerUser = 

    


# In[ ]:





# In[46]:


# as given by textbook ch4, jmcauley.
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return (numer / denom)


# In[47]:


# as given by textbook ch4, jmcauley.
def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2])
        #sim = Pearson(i, i2) # Could use alternate similarity metrics straightforwardly
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:10]


# In[ ]:





# In[48]:


answers['Q6'] = mostSimilar('2767052', 10)
answers['Q6']


# In[49]:


assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)


# In[50]:


### Question 7


# In[51]:


userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    rs = [ratingDict[(u,i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)
    
for i in usersPerItem:
    rs = [ratingDict[(u,i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)
    


# In[52]:


# Some data structures you might want
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataTest:
    user,item = d['user_id'], d['book_id']
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)


# In[53]:


ratingMean = sum([d['rating'] for d in dataTest]) / len(dataTest)
ratingMean


# In[54]:


#i2 = None
def predictRating(user,item): # src. chapter 4 notes predictRating()
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) == Jaccard(usersPerItem[item],usersPerItem[i2])):    # (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean


# In[55]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[56]:


#u,i = dataset[1]['user_id'], dataset[1]['book_id']


# In[57]:


# u,i = dataset[1]['user_id'], dataset[1]['book_id']
#predictRating(u, i)

# alwaysPredictMean = [ratingMean for d in dataset]
simPredictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]    # based on our predictRating() model
#print(simPredictions)
#print(predictRating(d['user_id'], d['book_id']) for d in dataset)

labels = [d['rating'] for d in dataTest]
#print(labels)

mse7 = MSE(simPredictions, labels)
mse7


# In[58]:


answers['Q7'] = mse7


# In[59]:


assertFloat(answers['Q7'])


# In[60]:


### Question 8


# In[61]:


#i2 = None
def predictRating(user,item): # src. chapter 4 notes predictRating()
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[i2],usersPerItem[item]))
    if (sum(similarities) == Jaccard(usersPerItem[item],usersPerItem[i2])):    # (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean


# In[62]:


simPredictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]    # based on our predictRating() model


labels = [d['rating'] for d in dataTest]
#print(labels)

mse8 = MSE(simPredictions, labels)
mse8


# In[63]:


answers['Q8'] = mse8


# In[64]:


assertFloat(answers['Q8'])


# In[65]:


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




