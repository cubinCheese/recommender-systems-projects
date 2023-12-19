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
import pandas
from sklearn import linear_model
from sklearn.metrics import accuracy_score


'''
In this assignment you will build recommender systems to make predictions related to book reviews from Goodreads.
Submissions will take the form of prediction files
'''


# In[2]:


# task 1: Read prediction 
# predict given (user,book) pair from pairs_Read.csv
# whether the user would read the book 
''' Go through pairs in pairs_Read.csv, check whether the user would read the book, mark as 1/0
    and then calculate accuracy. and place predictions into pairs_Read.csv'''
    
predictions_ReadCSV = open("predictions_Read.csv",'w')


predictions_ReadCSV.close()


# In[3]:


# Read file setup

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

        
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[4]:


# Data setup

allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
    
    

ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    

trainRatings = [r[2] for r in ratingsTrain]
globalAverage = sum(trainRatings) * 1.0 / len(trainRatings)



# Generate a negative set

userSet = set()
bookSet = set()
readSet = set()

for u,b,r in allRatings:
    userSet.add(u)
    bookSet.add(b)
    readSet.add((u,b))

lUserSet = list(userSet)
lBookSet = list(bookSet)

notRead = set()
for u,b,r in ratingsValid:
    #u = random.choice(lUserSet)
    b = random.choice(lBookSet)
    while ((u,b) in readSet or (u,b) in notRead):
        b = random.choice(lBookSet)
    notRead.add((u,b))

readValid = set()
for u,b,r in ratingsValid:
    readValid.add((u,b))


    
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()


# In[5]:



''' 
return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break    
'''    
    

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0


def CosineSet(s1, s2):
    # Not a proper implementation, operates on sets so correct for interactions only
    numer = len(s1.intersection(s2))
    denom = math.sqrt(len(s1)) * math.sqrt(len(s2))
    if denom == 0:
        return 0
    return numer / denom


'''
def Pearson(i1, i2):
    # Between two items
    iBar1 = itemAverages[i1]
    iBar2 = itemAverages[i2]
    inter = usersPerItem[i1].intersection(usersPerItem[i2])
    numer = 0
    denom1 = 0
    denom2 = 0
    for u in inter:
        numer += (ratingDict[(u,i1)] - iBar1)*(ratingDict[(u,i2)] - iBar2)
    for u in inter: #usersPerItem[i1]:
        denom1 += (ratingDict[(u,i1)] - iBar1)**2
    #for u in usersPerItem[i2]:
        denom2 += (ratingDict[(u,i2)] - iBar2)**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom
'''


# Improved strategy

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > 1.5 * totalRead/2: break #  1.5 * totalRead/2
        
        
'''
def baseline():
    # Evaluate baseline strategy

    correct = 0
    p0, p1 = 0,0
    for (label,sample) in [(1, readValid), (0, notRead)]:
        for (u,b) in sample:
            pred = 0
            if b in return1:
                pred = 1
            if pred == label:
                correct += 1
    print("baseline strategy: " + str(correct / (len(readValid) + len(notRead)))) # what we want to beat

baseline()

def baseline_modified():
    # Evaluate baseline strategy

    correct = 0
    p0, p1 = 0,0
    for (label,sample) in [(1, readValid), (0, notRead)]:
        for (u,b) in sample:
            pred = 0
            if b in return1:
                pred = 1
            if pred == label:
                correct += 1
    print("baseline strategy2: " + str(correct / (len(readValid) + len(notRead)))) # what we want to beat

baseline()'''


# Slow implementation, could easily be improved
'''
correct = 0
for (label,sample) in [(1, readValid), (0, notRead)]:
    for (u,b) in sample:
        maxSim = 0
        minSim = 0 # added
        users = set(ratingsPerItem[b])
        for b2,_ in ratingsPerUser[u]:
            sim = Jaccard(users,set(ratingsPerItem[b2]))
            if sim > maxSim:
                maxSim = sim
            if sim < minSim: # added
                minSim = sim # added
        pred = 0
        # maxSim > 0.013 or len(ratingsPerItem[b]) > 40:       # 0.74745
        # minSim > 0.013 or len(ratingsPerItem[b]) > 40:      # 0.7453
        # minSim < 0.013 or len(ratingsPerItem[b]) > 40:      # 0.75
        # minSim > 0.013 or len(ratingsPerItem[b2]) > 40:      # 0.5
        # minSim > 0.013 or len(ratingsPerUser[u]) > 40:      # 0.5
        # minSim > 0.013 or len(ratingsPerItem[u]) > 40:      # 0.5
        # minSim > 0.013 and len(ratingsPerItem[b]) > 40:      # 0.5
        # sim > 0.013 or len(ratingsPerItem[b]) > 40:      # 0.74585
        # minSim > 0.013 or b in return1 or len(ratingsPerItem[b]) > 40:      # 0.75605
        # maxSim > 0.013                       # 0.51035
        # maxSim > 0.013 or b in return1:      # 0.7558  
        # maxSim > 0.013 and b in return1:     # 0.51085
        # maxSim > 0.013 or b in return1 or len(ratingsPerItem[b]) > 40:        # 0.7551
        # maxSim > 0.013 or (b in return1 and len(ratingsPerItem[b]) > 40):     # 0.74805
        # maxSim > 0.013 or len(ratingsPerItem[b]) > 40:     # 0.74745
        if maxSim > 0.013 or b in return1 or len(ratingsPerItem[b]) > 40: # or len(ratingsPerItem[b]) > 40:      # 0.75605 # 0.75465
            pred = 1
        if pred == label:
            correct += 1

'''
''' 
correct = 0
for (label,sample) in [(1, readValid), (0, notRead)]:
    for (u,b) in sample:
        maxSim = 0
        minSim = 0 # added
        users = set(ratingsPerItem[b])
        for b2,_ in ratingsPerUser[u]:
            sim = Jaccard(users,set(ratingsPerItem[b2]))
            if sim > maxSim:
                maxSim = sim
            if sim < minSim: # added
                minSim = sim # added
        pred = 0
        # maxSim > 0.013 or len(ratingsPerItem[b]) > 40:       # 0.7648
        # minSim > 0.013 or len(ratingsPerItem[b]) > 40:      # 0.74505
        # minSim < 0.013 or len(ratingsPerItem[b]) > 40:      # 0.75
        # minSim > 0.013 or len(ratingsPerItem[b2]) > 40:      # 0.5
        # minSim > 0.013 or len(ratingsPerUser[u]) > 40:      # 0.5
        # minSim > 0.013 or len(ratingsPerItem[u]) > 40:      # 0.5
        # minSim > 0.013 and len(ratingsPerItem[b]) > 40:      # 0.5
        # sim > 0.013 or len(ratingsPerItem[b]) > 40:      # 0.74585
        # maxSim > 0.013                       # 0.51035
        # maxSim > 0.013 or b in return1:      # 0.7558  
        # maxSim > 0.013 and b in return1:     # 0.51085
        # maxSim > 0.013 or b in return1 or len(ratingsPerItem[b]) > 40:        # 0.7558
        # maxSim > 0.013 or (b in return1 and len(ratingsPerItem[b]) > 40):     # 0.74805
        minSim > 0.1 or len(ratingsPerItem[b]) > 40: # 0.7452
        if maxSim > 0.013 or len(ratingsPerItem[b]) > 40:     # 0.74805
            pred = 1
        if pred == label:
            correct += 1  
print("modified: " + str(correct / (len(readValid) + len(notRead))))
'''


# hw3-Q5 ---> actually writing to file
predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    maxSim = 0
    users = set(ratingsPerItem[b])
    for b2,_ in ratingsPerUser[u]:
        sim = Jaccard(users,set(ratingsPerItem[b2]))
        if sim > maxSim:
            maxSim = sim
    pred = 0
    if maxSim > 0.013 or b in return1 or len(ratingsPerItem[b]) > 40:
        pred = 1
    _ = predictions.write(u + ',' + b + ',' + str(pred) + '\n')

predictions.close()

print(" ")


# In[ ]:





# In[6]:


# category predictions


# In[9]:


# Same as previous solution, with larger dictionary

data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)
 
    
wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

# NW=1000 -> 2000 was an improvement.
NW = 10000 # dictionary size # represents NW most common words # since counts[] holds sorted most common words.


words = [x[1] for x in counts[:NW]]

wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

def feature(datum):
    feat = [0]*len(words)
    r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
    for w in r.split():
        if w in wordSet:
            feat[wordId[w]] += 1
    feat.append(1) #offset
    return feat



X = [feature(d) for d in data]
y = [d['genreID'] for d in data]

Xtrain = X[:9*len(X)//10]
ytrain = y[:9*len(y)//10]
Xvalid = X[9*len(X)//10:]
yvalid = y[9*len(y)//10:]


# In[ ]:


mod = linear_model.LogisticRegression(C=10,n_jobs=4,solver='lbfgs')
mod.fit(Xtrain, ytrain)
pred = mod.predict(Xvalid)
correct = pred == yvalid
sum(correct) / len(correct)


# In[ ]:


data_test = []
for d in readGz("test_Category.json.gz"):
    data_test.append(d)
Xtest = [feature(d) for d in data_test]
pred_test = mod.predict(Xtest)   
correct = pred_test == yvalid
sum(correct)/len(correct)


# In[ ]:


print(len(data))


# In[ ]:



      
predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    _ = predictions.write(u + ',' + b + ',' + str(pred_test[pos]) + '\n')
    pos += 1

predictions.close()    
    


# In[ ]:





# In[ ]:





# In[ ]:




