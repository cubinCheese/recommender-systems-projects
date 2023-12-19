#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model


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


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[5]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[6]:


answers = {}


# In[7]:


# Some data structures that will be useful


# In[8]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# In[9]:


len(allRatings)


# In[10]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# In[11]:


##################################################
# Rating prediction (CSE258 only)                #
##################################################


# In[ ]:





# In[12]:


### Question 9


# In[ ]:





# In[13]:


answers['Q9'] = validMSE


# In[ ]:


assertFloat(answers['Q9'])


# In[14]:


### Question 10


# In[15]:


answers['Q10'] = [maxUser, minUser, maxBeta, minBeta]


# In[16]:


assert [type(x) for x in answers['Q10']] == [str, str, float, float]


# In[17]:


### Question 11


# In[ ]:





# In[18]:


answers['Q11'] = (lamb, validMSE)


# In[19]:


assertFloat(answers['Q11'][0])
assertFloat(answers['Q11'][1])


# In[20]:


predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # (etc.)
    
predictions.close()


# In[21]:


##################################################
# Read prediction                                #
##################################################


# In[22]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break


# In[23]:


### Question 1


# In[24]:


# NOTE. we don't care about actual ratings in q1-4. 
# we can replace that rating value column with read/not read


# In[25]:


def accuracy(predictions,y):
    correct = (predictions == y)*1
    return sum(correct) / len(correct)


# In[26]:


# create list of all books (each element will be a unique book_id)
allBooks = set()
for u,b,r in allRatings:
    allBooks.add(b)
allBooks

type(allBooks)
allBooks = list(allBooks)
type(allBooks)


# In[27]:


# create list of negative samples (books a user hasn't read) 

negSamples = []#set() 
for u,b,r in ratingsValid: # only going over ratingsValid results in only positives
    book = random.choice(allBooks) # choose exactly 1 book user hasn't read
    while book in ratingsPerUser[u]: # while book is in ratingsPerUser 
        book = random.choice(allBooks)  # make a new book choice
    negSamples.append((u,book,0))   # append book
negSamples
# len(negSamples) # correct size of 10000
# negSamples = list(set(negSamples)) # get rid of duplicates
len(negSamples)


# In[28]:


ratingsValid_withoutRating = [(u,b,1) for u,b,r in ratingsValid]

# create new validation set # where 10000: are pos, :10000 are neg
newValidationSet = ratingsValid_withoutRating + negSamples # piazza: add ratingsValid ontop of negSamples
# newvalidationset should be (userID, bookID, read/not = 1/0)
len(newValidationSet) # correct size of 20000


# In[29]:


# newValidationSet[9999] = ('u55163096', 'b31362596', 4)
# newValidationSet[10000] = ['u59070515', 'b49433911']
newValidationSet[9999]
newValidationSet[10000]


# In[30]:


# calculate predictions: go through validiation set, for each book,
# predict read: if it's in return1, and not read otherwise
pred = []
for u,b,r in newValidationSet: # newValidationSet
    if (b in return1): # check if book in validationset is in return1
        #pred.append('true') # 
        pred.append(1)
    else: pred.append(0)
    #else: pred.append('false')
    
pred
len(pred)


# In[31]:


# accuracy formula takes two parameters of 1-D
# need to change newValidationSet to be on list of read/not read (1/0s)
yLabel = []
for u,b,r in newValidationSet:
    yLabel.append(r) # append read/not read
    
print(yLabel[1],pred[1])
#print(yLabel)
# acc1 = len(return1) / len(pred)
# acc1 = len(return1) / len(pred) # len(pred)/len(return1)
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(pred,yLabel) # return1 has user_id, while pred has 1/0??
acc1


# In[ ]:





# In[32]:


answers['Q1'] = acc1


# In[33]:


assertFloat(answers['Q1'])


# In[34]:


### Question 2


# In[ ]:





# In[65]:


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return2 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return2.add(i)
    if count > totalRead/4: break # changed


# In[66]:


# re-calculate predictions: go through validiation set, for each book,
# predict read: if it's in return1, and not read otherwise
pred2 = []
for u,b,r in newValidationSet: # newValidationSet
    if (b in return2): # check if book in validationset is in return1
        pred2.append(1)
    else: pred2.append(0)
    #else: pred.append('false')
    
pred2
len(pred2)


# In[ ]:





# In[67]:


yLabel = []
for u,b,r in newValidationSet:
    yLabel.append(r) # append read/not read
    


# In[68]:


threshold = 1/4
# acc2 = sum(pred2) / len(yLabel) # rerun q1 with 3/4 threshold having modified return1
acc2 = accuracy_score(pred2,yLabel)


# In[ ]:





# In[69]:


answers['Q2'] = [threshold, acc2]


# In[70]:


assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])


# In[71]:


### Question 3/4


# In[72]:


# only dealing with books having been read (original ratingsValid)


# In[73]:


# isolate training data to exclude ratings
ratingsTrain_ub = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsTrain_ub[u].append(b)
    
ratingsTrain_bu = defaultdict(list) # inverted key,value pairs
for u,b,r in ratingsTrain:
    ratingsTrain_bu[b].append(u)


# In[74]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[75]:


# re-creating new validation set we just built, just without ratings - easier access
newValidationSet_withoutRatings = [(u,b) for u,b,r in ratingsValid] + [(u,b) for u,b,r in negSamples]


# In[79]:


def JaccardPrediction():  # predict using jaccard
    pred=[]
    similarities = []
    for u,b in newValidationSet_withoutRatings:
        otherBooksU = ratingsTrain_ub[u] 
        similarity = 0
        # compute similarity of particular b and b'
        for books in otherBooksU:
            firstSet = set(ratingsTrain_bu[b])
            secondSet = set(ratingsTrain_bu[books])
            similarity = Jaccard(firstSet,secondSet) # compute Jaccard similarity
            similarities.append(similarity)          # append instance similarity to list
        if similarity > 1/4:
            pred.append(1)
        else: pred.append(0)
        #len(pred)
    return pred


# In[80]:


predJacc = JaccardPrediction()
acc3 = accuracy_score(predJacc,yLabel)
acc3


# In[81]:


def Jaccard_ThresholdPrediction():  # predict using jaccard
    pred=[]
    similarities = []
    for u,b in newValidationSet_withoutRatings:
        bookList = ratingsTrain_ub[u] # list of books for user
        similarity = 0
        # compute similarity of particular b and b'
        for books in bookList:
            firstSet = set(ratingsTrain_bu[b])
            secondSet = set(ratingsTrain_bu[books])
            similarity = Jaccard(firstSet,secondSet) # compute Jaccard similarity
            similarities.append(similarity)          # append instance similarity to list
        if (similarity > 1/4) and (b in return1):
            pred.append(1)
        else: pred.append(0)
        #len(pred)
    return pred

predJaccThresh = Jaccard_ThresholdPrediction()
acc4 = accuracy_score(predJaccThresh,yLabel)
acc4


# In[82]:


answers['Q3'] = acc3 
answers['Q4'] = acc4
assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[ ]:





# In[ ]:





# In[118]:


def readPredictions_q5():
    pred_q5 = []
    similarities = []
    predictions = open("predictions_Read.csv", 'w')
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        bookList = ratingsTrain_ub[u]      # books that user read
        similarity = 0
        for books in bookList:                       # for each book
            firstSet = set(ratingsTrain_bu[b])
            secondSet = set(ratingsTrain_bu[books])

            similarity = Jaccard(firstSet,secondSet) # compute Jaccard similarity
            similarities.append(similarity)          # append instance similarity to list

        if (similarity > 0.05) and (b in return1):
            pred_q5.append(1)
            predictions.write(u + ',' + b + ",1\n")
        else: 
            pred_q5.append(0)
            predictions.write(u + ',' + b + ",0\n")
    predictions.close()
    return pred_q5


# In[119]:


pred_q5 = readPredictions_q5()
acc5 = accuracy_score(pred_q5,yLabel)
acc5


# In[87]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[88]:


assert type(answers['Q5']) == str


# In[89]:


##################################################
# Category prediction (CSE158 only)              #
##################################################


# In[90]:


### Question 6


# In[91]:


data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)


# In[92]:


data[0]


# In[93]:


trainReviews = data[90000:]
validReviews = data[:10000]


# In[94]:


validReviews


# In[95]:


import re  
import itertools 

def feat_q6(): # present common words
    
    # (1) in training set
    # remove punctuation
    # remove capitalization
    # find 1000 most common words across all reviews ('review_text') 
    for entry in trainReviews:
        reviewTextStr = entry['review_text']
        reviewTextStr = re.sub(r'[^\w\s]', '', reviewTextStr) # remove punctuation
        reviewTextStr = reviewTextStr.lower() # remove capitalizations
        entry['review_text'] = reviewTextStr  # replace str
        # print(entry)
    
    # create dict
    freqDict = {} # freq table
    for entry in trainReviews:
        listOfWords = [] # list of words in particular entry's review_text
        targetString = entry['review_text']
        listOfWords = targetString.split()
        # print(listOfWords)
        for word in listOfWords:
            if (word not in freqDict):
                freqDict[word] = 1
            elif (word in freqDict):
                freqDict[word] += 1 # increment count of word
            else:
                print("error: should never reach here")
                exit(1)
    #print(freqDict)
    
    # now go through freq table and find 1000 highest count words
    mostCommonWords_1000 = {}
    # sort dictionary by values (count of words) - from greatest to least 
    reSort_freqDict = {key: val for key, val in sorted(freqDict.items(), key = lambda ele: ele[1], reverse = True)}
    #print(reSort_freqDict)
    mostCommonWords_1000 = dict(itertools.islice(reSort_freqDict.items(), 1000)) # slice dict to get 1000 most common words
    #print(len(mostCommonWords_10000))
    #print(mostCommonWords_10000)
    return mostCommonWords_1000
#feat_q6()


# In[96]:


# could have just used lecture notes:
'''
def feat_q6_fromLec(data):
    wordCount = defaultdict(int)
    punctuation = set(string.punctuation)
    for d in data: # Strictly, should just use the *training* data to extract word counts
        r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
        for w in r.split():
            wordCount[w] += 1
    
    wordCount = {key: val for key, val in sorted(wordCount.items(), key = lambda ele: ele[1], reverse = True)}

    return wordCount
feat_q6_fromLec(trainReviews)'''


# In[97]:


# report top 10 most common words
mostCommonWords_1000 = feat_q6()
counts = [] # need to report as list of tuples (freq,word)
mostCommonWords_10 = dict(itertools.islice(mostCommonWords_1000.items(), 10)) # slice dict to get 10 most common words
for key,value in mostCommonWords_10.items():
    counts.append((value,key)) # make sure to flip into freq,word
len(counts)
print(counts)


# In[98]:


mostCommonWords_1000


# In[99]:


answers['Q6'] = counts[:10]


# In[100]:


assert [type(x[0]) for x in answers['Q6']] == [int]*10
assert [type(x[1]) for x in answers['Q6']] == [str]*10


# In[101]:


### Question 7


# In[102]:


newCounts = []
for key,value in mostCommonWords_1000.items():
    newCounts.append((value,key)) # make sure to flip into freq,word

words = [x[1] for x in newCounts[:1000]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

def feature(datum):
    feat = [0]*len(words)
    punctuation = set(string.punctuation)
    r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
    for w in r.split():
        if w in words:
            feat[wordId[w]] += 1
    feat.append(1) # offset
    return feat


# In[103]:



X = [feature(d) for d in trainReviews]
y = [reviews['genreID'] for reviews in trainReviews] # set y (aka labels)


Xtrain = X[:9*len(X)//10] # doesn't get used in feature()
ytrain = y[:9*len(y)//10]
Xvalid = X[9*len(X)//10:]
yvalid = y[9*len(y)//10:]


# In[104]:


from sklearn.metrics import accuracy_score

model = linear_model.LogisticRegression(fit_intercept=False)
model.fit(Xtrain, ytrain)
predictions = model.predict(Xvalid)
# correct = predictions == ytrain
# correct = sum([p==l]for (p,l)in zip(predictions,yvalid))
acc7 = accuracy_score(predictions,yvalid) #sum(correct) / len(yvalid)
acc7


# In[ ]:





# In[105]:


answers['Q7'] = acc7


# In[106]:


assertFloat(answers['Q7'])


# In[107]:


### Question 8


# In[108]:


from sklearn.metrics import accuracy_score

modelq8 = linear_model.LogisticRegression(C=1000)
modelq8.fit(Xtrain, ytrain)
pred_q8 = modelq8.predict(Xvalid)
# correct = predictions == ytrain
# correct = sum([p==l]for (p,l)in zip(predictions,yvalid))
acc8 = accuracy_score(pred_q8,yvalid) #sum(correct) / len(yvalid)
acc8


# In[109]:


answers['Q8'] = acc8


# In[110]:


assertFloat(answers['Q8'])


# In[111]:


# Run on test set


# In[112]:


predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    # (etc.)

predictions.close()


# In[113]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




