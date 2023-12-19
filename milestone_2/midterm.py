#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import gzip
import math
from collections import defaultdict
import numpy
from sklearn import linear_model


# In[4]:


# This will suppress any warnings, comment out if you'd like to preserve them
import warnings
warnings.filterwarnings("ignore")


# In[5]:


# Check formatting of submissions
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[6]:


answers = {}


# In[7]:


f = open("spoilers.json.gz", 'r')


# In[8]:


dataset = []
for l in f:
    d = eval(l)
    dataset.append(d)


# In[9]:


f.close()


# In[10]:


# A few utility data structures
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['user_id'],d['book_id']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)

# Sort reviews per user by timestamp
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['timestamp'])

# Same for reviews per item
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['timestamp'])


# In[11]:


# E.g. reviews for this user are sorted from earliest to most recent
[d['timestamp'] for d in reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e']]


# In[12]:


### 1a


# In[ ]:





# In[170]:


# Model 1 (a): for each user, predict last rating based on avg of previous ratings


# In[171]:


def predictUser(currReview):
    total_ratings = 0
    i = 0
    for j in range(0,len(currReview)-1): # loop through all reviews of user
        total_ratings = total_ratings + currReview[j]['rating']
        i+=1
    # final predicted rating for particular user
    res = (total_ratings/i)
    return res
    
def predictUser_lastRating():
    pred = []
    y = []
    for reviews in reviewsPerUser: # for each review in reviewsPerUser
        # number of reviews the particular user has provided in total
        totalReviewCount = len(reviewsPerUser[reviews]) # of a particular user
        if (totalReviewCount > 1): # must have atleast 
            userPrediction = predictUser(reviewsPerUser[reviews])
            pred.append(userPrediction)
            y.append(reviewsPerUser[reviews][-1]['rating'])
    return (y,pred)

(y,pred) = predictUser_lastRating()


# In[172]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[173]:


mse = MSE(pred,y)
mse


# In[ ]:





# In[174]:


# average of an user's previous ratings


# Predicts most recent rating based on user's 


# In[ ]:





# In[ ]:





# In[ ]:





# In[175]:


#simPredictions = [predictRating(d['user_id'], d['book_id']) for d in dataset]    # based on our predictRating() model
#labels = [d['rating'] for d in dataset]
#mse = MSE(simPredictions, labels)
#mse


# In[ ]:





# In[ ]:





# In[176]:


answers['Q1a'] = MSE(y,pred)


# In[177]:


assertFloat(answers['Q1a'])


# In[178]:


### 1b


# In[ ]:





# In[179]:


def predictItem(currReview):
    total_ratings = 0
    i = 0
    for j in range(0,len(currReview)-1): # loop through all reviews of user
        total_ratings = total_ratings + currReview[j]['rating']
        i+=1
    # final predicted rating for particular user
    res = (total_ratings/i)
    return res
    
def predictItem_lastRating():
    pred = []
    y = []
    for reviews in reviewsPerItem: # for each review in reviewsPerUser
        # number of reviews the particular user has provided in total
        totalReviewCount = len(reviewsPerItem[reviews]) # of a particular user
        if (totalReviewCount > 1): # must have atleast 1 review
            userPrediction = predictUser(reviewsPerItem[reviews])
            pred.append(userPrediction)
            y.append(reviewsPerItem[reviews][-1]['rating'])
    return (y,pred)

y1,pred1 = predictItem_lastRating()


# In[180]:


mse = MSE(pred1,y1)
mse


# In[ ]:





# In[181]:


answers['Q1b'] = MSE(y1,pred1)


# In[182]:


assertFloat(answers['Q1b'])


# In[183]:


### 2


# In[ ]:





# In[184]:


# modifying previous predictUser() from q1, now we incorporate N-ratings
def predictUser_q2(currReview,N):
    total_ratings = 0
    
    i = 0
    for j in range(len(currReview)-1-N,len(currReview)-1): # loop through all reviews of user
        total_ratings = total_ratings + currReview[j]['rating']
        i+=1
        
    # final predicted rating for particular user
    res = (total_ratings/i)
    return res
    
def predictUser_last_Nratings(N):
    pred = []
    y = []
        
    for reviews in reviewsPerUser: # for each review in reviewsPerUser
        
        # number of reviews the particular user has provided in total
        totalReviewCount = len(reviewsPerUser[reviews]) # of a particular user

        # case: discard instances with only a single rating
        if totalReviewCount == 1:
            continue
            
        # case: fewer than N+1 ratings
        # truncate - take avg over currently avaliable ratings
        elif totalReviewCount < N+1: # case: fewer than N+1 ratings
            userPrediction = predictUser(reviewsPerUser[reviews])
            pred.append(userPrediction)
            y.append(reviewsPerUser[reviews][-1]['rating'])
            
        # case: otherwise, we compute based on last N-ratings    
        else:
            userPrediction = predictUser_q2(reviewsPerUser[reviews],N)
            pred.append(userPrediction)
            y.append(reviewsPerUser[reviews][-1]['rating'])
        
    return (y,pred)


# In[ ]:





# In[185]:


answers['Q2'] = []

for N in [1,2,3]:
    y2,pred2 = predictUser_last_Nratings(N)
    # etc.
    answers['Q2'].append(MSE(y2,pred2))


# In[186]:


assertFloatList(answers['Q2'], 3)
answers['Q2']


# In[ ]:





# In[187]:


print(reviewsPerUser)


# In[188]:


### 3a


# In[189]:


from collections import OrderedDict

def revDict(dict1):
    return OrderedDict(reversed(dict1.items()))


# In[190]:


# reviewsPerUser[u] = gives data of all reviews given by particular user.


# In[ ]:





# In[191]:


def feature3(N, u):
    feat = [1]
    ctr = 0        # counts to N # so we can break after using last N ratings for calc.
    for i in reversed(reviewsPerUser[u]):
        if (ctr == N+1): # break when we reach Nth rating
            break      # since (N+1)th rating is what we want to predict
        elif (ctr != 0):
            feat.insert(len(feat), i['rating']) #
        ctr += 1
    return feat


# In[192]:


#reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e'][0]['rating']
reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e']


# In[193]:


answers['Q3a'] = [feature3(2,dataset[0]['user_id']), feature3(3,dataset[0]['user_id'])]


# In[194]:


answers['Q3a']


# In[195]:


assert len(answers['Q3a']) == 2
assert len(answers['Q3a'][0]) == 3
assert len(answers['Q3a'][1]) == 4


# In[196]:


[d['rating'] for d in reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e']]


# In[ ]:





# In[197]:


# test cell
testrev = revDict(reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e'][0]) # is within a list of dictionaries, need to pull out
testrev


# In[198]:


[d['timestamp'] for d in reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e']]


# In[199]:


print(reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e'])


# In[200]:


### 3b


# In[201]:


# calculate the MSE for window sizes N=1, N=2, N=3
# predict current rating based on previous N=1 ratings,
# predict current rating based on previous N=2 ratings, etc...
    # trying to predict the last rating (current) of an item
# re-thinking the problem, in isolation of 3a.


# In[202]:


# returns feature vector of ratings value at particular user u, constrained by last N ratings
def feature_helper(N,u): 
    featHelper = [1]   # feature vector to return
    totalReviewCount = len(reviewsPerUser[u])  # number of reviews made by user
    if (totalReviewCount == 2):                # user has 2 rating reviews only
        return [1,reviewsPerUser[u][0]['rating']] # we always discard the rating we're predicting for
                                                  # hence, we end up returning the only 'historical' rating user made.
    # reversing through N-ratings
    for i in range(totalReviewCount-1-N,totalReviewCount-1):
        featHelper.append(reviewsPerUser[u][i]['rating']) # appending ratings value to feature vector
    return featHelper


# In[203]:


def feature3b(N):
    feat = []
    y = []
    
    # loop through reviews in reviewsPerUser
    for review in reviewsPerUser:
        
        totalReviewCount = len(reviewsPerUser[review])
        # if user has no historical ratings -- i.e. exactly one rating review
        if (totalReviewCount == 1):
            continue
        
        # if More N windows than total review/ratings made by user
        elif (totalReviewCount < N+1):
            continue
        else:
            # retrieve and append feature vector of last N-ratings
            featToAppend = []
            featToAppend = feature_helper(N,reviewsPerUser[review][0]['user_id'])
            feat.append(featToAppend)
            
            # if there're only 2 reviews for a user, just append non-historical review
            if (totalReviewCount == 2):
                yVal = reviewsPerUser[review][1]['rating']
                y.append(yVal)
            else: # otherwise, we have more than 2 reviews for a user. Append y-vals for model.
                yVal = reviewsPerUser[review][-1]['rating']
                y.append(yVal)
    return (feat,y)


# In[204]:


[reviewsPerUser[review][0]['rating'] for review in reviewsPerUser]


# In[205]:


answers['Q3b'] = []

# for each N-window size of N=1, N=2, N=3
for N in [1,2,3]:
    (feat,y) = feature3b(N) # compute feature vector and y
    X = numpy.matrix(feat)  # convert feature vector to matrix
    
    # use linear regression model
    mod = linear_model.LinearRegression(fit_intercept=False)
    mod.fit(X,y)           # fit the model
    
    yPred = mod.predict(X) # model prediction
    
    # computing mse
    sse = sum([r**2 for r in y-yPred]) # compute sum-squared-error
    mse = (sse / len(y))               # compute mean-squared-error
    
    answers['Q3b'].append(mse)         # add N's mse answer
    
answers['Q3b']


# In[206]:


assertFloatList(answers['Q3b'], 3)


# In[207]:


### 4a


# In[208]:


globalAverage = [d['rating'] for d in dataset]
globalAverage = sum(globalAverage) / len(globalAverage)


# In[209]:


def featureMeanValue(N, u): # For a user u and a window size of N
    feat = [1]
    j=0
    '''
    # isolate last N ratings in reviewsPerUser[u] 
    last_NRatings = []
    ctr = 0
    for reviews in reviewsPerUser[u]: # loop through user ratings
        last_NRatings.append(reviews['rating'][ctr]) # list of ratings
        ctr += 1
    last_NRatings = last_NRatings[-N:] # get last N-ratings
        
    # 
    for rating in last_NRatings:
        if (rating == reviewsPerUser[u])
        
   
    for i in range(0,N):
        if (reviewsPerUser[u][-1]['rating'] == reviewsPerUser[u][i]['rating']): # discard rating we're trying to compute
            continue # i.e. we don't put into list
        if (reviewsPerUser[u][i]['rating'] not in reviewsPerUser[u]):
            # sum of all ratings / number of ratings
            average = sum(reviewsPerUser[u]) / len(reviewsPerUser[u])
            feat.append(average)
        elif (len(reviewsPerUser[u]) < 1): # no historical ratings
            feat.append(globalAverage)
        j += 1
    return feat
    '''


# In[19]:


def featureMeanValue(N, u): # For a user u and a window size of N
    feat = [1]
    count=0
    sumRatings = 0
    
    # no historical ratings case
    if len(reviewsPerUser[u]) < 1:
        i = 0
        while (i<N):
            feat.append(globalAverage)
            i+=1
        return feat
    
    # reversed list of N-ratings
    rev_lastNRatings = []
    for review in reversed(reviewsPerUser[u]):
        rev_lastNRatings.append(review['rating'])
    
    
    # dealing with discard and each missing index case
    for ratings in rev_lastNRatings:
        if (count > N+1): # N+1 th rating we're trying to compute
            break     # discard
        elif (count!=N) and (count!=0):     # if still have NONEs and historical rating exists
            feat.append(ratings) # append ratings to feature
            sumRatings += ratings # compute sum of ratings
        count+=1 #count++
        
    # if we still have NONEs within feature vector
    if (count!=N+1):
        # for remaining slots in feature vector that were NONE
        for i in range(0, N-count+1):
            average = sumRatings/count # compute user's average rating value
            feat.append(average)   # replace missing index with average
    
    return feat
    


# In[20]:


featureMeanValue(10, dataset[0]['user_id'])


# In[21]:


reviewsPerUser['50c6444411eeced02e89173941e6f2f2'][0]['rating']
reviewsPerUser['50c6444411eeced02e89173941e6f2f2'][-1]['rating']
#len(reviewsPerUser['50c6444411eeced02e89173941e6f2f2']) # number of reviews provided by user


# In[22]:


def featureMissingValue(N, u):
    feat = [1]
    
    #for i in range(0,N):
        #if (feat is missing)
    # return garbage - temp
    #return ([1. , 4. , 4. , 4. , 4. , 5. , 3.5, 3.5, 3.5, 3.5, 3.5, 1. , 4. , 4. , 4. , 5. , 3.5, 3.5, 3.5, 3.5, 3.5])


# In[23]:


answers['Q4a'] = [featureMeanValue(10, dataset[0]['user_id']), featureMissingValue(10, dataset[0]['user_id'])]


# In[289]:


assert len(answers['Q4a']) == 2
assert len(answers['Q4a'][0]) == 11
#assert len(answers['Q4a'][1]) == 21


# In[270]:


### 4b


# In[271]:


answers['Q4b'] = []

for featFunc in [featureMeanValue, featureMissingValue]:
    # etc.
    answers['Q4b'].append(mse)


# In[272]:


assertFloatList(answers["Q4b"], 2)


# In[273]:


### 5


# In[220]:



def feature5(sentence):
    feat = [1]
    feat.append(len(sentence))       # length in chars (length of full review)
    feat.append(sentence.count('!')) # number of "!"  # within a [[]]
    numUpperInStr = sum(1 for char in sentence if char.isupper())
    feat.append(numUpperInStr)      # number of capital letters
    return feat


# In[221]:


y = [] # list of integers representing spoilers
X = [] # list of strings representing sentences
# parallel lists

for d in dataset:
    for spoiler,sentence in d['review_sentences']: # (int,string)
        X.append(feature5(sentence)) # f(sentence contains spoiler)
        y.append(spoiler)            # theta0 
        


# In[222]:


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)


# In[223]:


answers['Q5a'] = X[0]
answers['Q5a']


# In[ ]:





# In[224]:


predictions = mod.predict(X)


TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])


TPR = TP / (TP + FN)
TNR = TN / (TN + FP)

BER = 1 - (1/2) * (TPR + TNR)


# In[ ]:





# In[225]:


answers['Q5b'] = [TP, TN, FP, FN, BER]
answers['Q5b']


# In[226]:


assert len(answers['Q5a']) == 4
assertFloatList(answers['Q5b'], 5)


# In[227]:


### 6


# In[228]:


def feature6(review):
    feat = [1]
    feat.append(len(sentence))       # length in chars (length of full review)
    feat.append(sentence.count('!')) # number of "!"  # within a [[]]
    numUpperInStr = sum(1 for char in sentence if char.isupper())
    feat.append(numUpperInStr)      # number of capital letters
    for i in range(0,5): # N=5 # for the last 5 sentences of a review
        #for i in reversed(review):
        #feat.append(len(review)) # length of review
        #feat.append(spoiler)
        feat.append(review['review_sentences'][i][0])
    return feat # predict spoilder label for 6th sentence


# In[229]:


y = []
X = []

for d in dataset:
    sentences = d['review_sentences']
    if len(sentences) < 6: continue
    X.append(feature6(d))
    y.append(sentences[5][0])
    
#etc.


# In[230]:


mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

predictions = mod.predict(X)

def BER(predictions,y):
    TP = sum([(p and l) for (p,l) in zip(predictions, y)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, y)])


    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    BER = 1 - (1/2) * (TPR + TNR)
    
    return BER

ber = BER(predictions,y)
ber


# In[231]:


answers['Q6a'] = X[0]
answers['Q6a'] 


# In[232]:


answers['Q6b'] = ber # BER


# In[233]:


assert len(answers['Q6a']) == 9
assertFloat(answers['Q6b'])


# In[234]:


### 7


# In[235]:


# 50/25/25% train/valid/test split
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[236]:


for c in [0.01, 0.1, 1, 10, 100]:
    # etc.


# In[298]:


C = [0.01, 0.1, 1, 10, 100]
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
berList


# In[293]:


min(berList)


# In[294]:


bestBER = min(berList) # smallest value of BER # lower BER is ideal
bestC = 100            # corresponds to 0.11332802789146579

bers = berList

# now, calculate BER on the test set w/ this C-value

mod_test = linear_model.LogisticRegression(C=100, class_weight='balanced') # using C=100
mod_test.fit(Xtest,ytest)

pred_test = mod_test.predict(Xtest)

# calculate BER
ber = BER(pred_test,ytest) 


# In[295]:


ber


# In[296]:


answers['Q7'] = bers + [bestC] + [ber]
answers['Q7'] 


# In[242]:


assertFloatList(answers['Q7'], 7)


# In[243]:


### 8


# In[244]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[245]:


# 75/25% train/test split
dataTrain = dataset[:15000]
dataTest = dataset[15000:]


# In[246]:


# A few utilities

itemAverages = defaultdict(list)
ratingMean = []

for d in dataTrain:
    itemAverages[d['book_id']].append(d['rating'])
    ratingMean.append(d['rating'])

for i in itemAverages:
    itemAverages[i] = sum(itemAverages[i]) / len(itemAverages[i])

ratingMean = sum(ratingMean) / len(ratingMean)


# In[247]:


reviewsPerUser = defaultdict(list)
usersPerItem = defaultdict(set)

for d in dataTrain:
    u,i = d['user_id'], d['book_id']
    reviewsPerUser[u].append(d)
    usersPerItem[i].add(u)


# In[248]:


# From my HW2 solution, welcome to reuse
def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            return ratingMean


# In[249]:


simPredictions = [predictRating(d['user_id'], d['book_id']) for d in dataTest]    # based on our predictRating() model
labels = [d['rating'] for d in dataTest] # ref to hw2

mse = MSE(simPredictions, labels)
mse


# In[250]:


# answers["Q8"] = MSE(predictions, labels)
answers["Q8"] = mse


# In[251]:


assertFloat(answers["Q8"])


# In[252]:


### 9


# In[ ]:





# In[253]:


# split dataTest based on a/b/c instances
dataTest_0 = []
dataTest_1to5 = []
dataTest_5plus = []

bookIDs = []

# isolate list of book IDs from training data
#[bookIDs.append(d['book_id']) for d in dataTrain]
for d in dataTrain:
    bookIDs.append(d['book_id'])

# loop through dataTest
for d in dataTest: #  u,i = d['user_id'], d['book_id']  # "i" just refers to particular book_id data
    
    appearanceCount = bookIDs.count(d['book_id']) # times that particular book appeared

    # instances where i appeared 0-times in training set
    if (d['book_id'] not in bookIDs):  # if particular book didn't appear
        dataTest_0.append(d)           # add it to corresponding array
    # instances where i appeared 1 to 5 times in training set
    elif (appearanceCount >= 1 and appearanceCount <= 5):
        dataTest_1to5.append(d)
    # instances where i appeared more than 5 times in training set
    elif (appearanceCount > 5):
        dataTest_5plus.append(d)

# print(appearanceCount)
print(len(dataTest))
print(len(dataTest_0))
print(len(dataTest_1to5))
print(len(dataTest_5plus))


# In[254]:


# calc mse for each subset
simPredictions_q9a = [predictRating(d['user_id'], d['book_id']) for d in dataTest_0]    # based on our predictRating() model
labels_q9a = [d['rating'] for d in dataTest_0] # ref to hw2

simPredictions_q9b = [predictRating(d['user_id'], d['book_id']) for d in dataTest_1to5]    # based on our predictRating() model
labels_q9b = [d['rating'] for d in dataTest_1to5] # ref to hw2

simPredictions_q9c = [predictRating(d['user_id'], d['book_id']) for d in dataTest_5plus]    # based on our predictRating() model
labels_q9c = [d['rating'] for d in dataTest_5plus] # ref to hw2

mse0 = MSE(simPredictions_q9a, labels_q9a)
mse1to5 = MSE(simPredictions_q9b, labels_q9b)
mse5 = MSE(simPredictions_q9c, labels_q9c)


# In[255]:


answers["Q9"] = [mse0, mse1to5, mse5]
answers["Q9"]


# In[256]:


assertFloatList(answers["Q9"], 3)


# In[257]:


### 10


# In[258]:


# task: modify q9's predictRating()
# goal: lower the MSE

def predictRating_q10(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            return ratingMean


# In[259]:


# originally dataTest, shifted to dataset. better MSE!
simPredictions = [predictRating_q10(d['user_id'], d['book_id']) for d in dataset]  
labels = [d['rating'] for d in dataset] # ref to hw2
#predictionsRounded = [int(p + 0.25) for p in simPredictions] # provides worse mse
itsMSE = MSE(simPredictions, labels)
itsMSE


# In[260]:


answers["Q10"] = ("I modified the data with which we call predictRating() in question 10. Instead of having modeling based on dataTest, I shifted the basis to the entire dataset. The logic being that the more data our predictior has to work off of, the less greater its prediction accuracy.", itsMSE)


# In[261]:


assert type(answers["Q10"][0]) == str
assertFloat(answers["Q10"][1])


# In[290]:


f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:





# In[ ]:




