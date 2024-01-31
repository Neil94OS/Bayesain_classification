#Neil O'Sullivan
#R00206266
#SDH4-C
import math
import numpy as np
import pandas as pd
from sklearn import svm, metrics

from sklearn.model_selection import StratifiedKFold # Read in csv file

data = pd.read_excel("movie_reviews.xlsx")

training_data = []
training_data_reviews = []
test_data = []
test_data_reviews = []
training_labels = []
test_labels = []
number_positive_training = 0
number_negative_training = 0
number_positive_test = 0
number_negative_test = 0
task3words=[]


def task1():
    data["Split"] = data["Split"].map(lambda x: 1 if x == "test" else 0) #Map split 1=Test, 0 = Training

    training_data_reviews = data[data["Split"] == 0][["Review"]]
    training_labels = data[data["Split"] == 0][["Sentiment"]]
    test_data_reviews = data[data["Split"] == 1][["Review"]]
    test_labels = data[data["Split"] == 1][["Sentiment"]]

    test_data = data[data["Split"] == 1].copy(deep=True)#Deep copies to slice again
    training_data = data[data["Split"] == 0].copy(deep=True)

    training_data["Sentiment"] = training_data["Sentiment"].map(lambda x: 1 if x == "positive" else 0) #map positive or negative to 1 or 0

    number_positive_training = training_data[training_data["Sentiment"] == 1][["Sentiment"]].count() # count each in the training set where sentiment = 1 (positive)
    number_negative_training = training_data[training_data["Sentiment"] == 0][["Sentiment"]].count() # count each in the training set where sentiment = 0 (negative)

    test_data["Sentiment"] = test_data["Sentiment"].map(lambda x: 1 if x == "positive" else 0) #map positive or negative to 1 or 0
    number_positive_test = test_data[test_data["Sentiment"] == 1][["Sentiment"]].count() # count each in the test set where sentiment = 1 (positive)
    number_negative_test = test_data[test_data["Sentiment"] == 0][["Sentiment"]].count() # count each in the test set where sentiment = 0 (negative)

    #Outputs as per brief
    print("Number of positive reviews in the training set: ",number_positive_training)
    print("Number of negative reviews in the training set: ",number_negative_training)
    print("Number of positive reviews in the test set: ",number_positive_test)
    print("Number of negative reviews in the test set: ",number_negative_test)



    return training_data_reviews, training_labels, test_data_reviews, test_labels # returns 4 lists as per brief

def task2(training_data_reviews, minWordLength, minWordOccurence):

    training_data_reviews = training_data_reviews['Review'].str.replace('[^a-zA-Z0-9]', ' ', regex=True).str.strip()        #Remove special characters
    training_data_reviews = training_data_reviews.str.lower()           #Change to lower case
    all_words = training_data_reviews.str.split()                       #Split into individual words

    new_words = []    #array for every word
    for i in all_words:  #for every word in all_words add to the new array
        new_words += i

    wordOccurences = {}       #dictionary for word occurrences
    words_result = []         #array for final result to be returned

    for word in new_words:                                              #Count occurrences of each word
        if (len(word) >= minWordLength):                                #if the length is equal to or above the minimum length input
            if (word in wordOccurences):
                wordOccurences[word] = wordOccurences[word] + 1         #if already in occurrences add to count
            else:
                wordOccurences[word] = 1                                #if not already in occurrences add word occurrences

    for word in wordOccurences:                                         #for every word in the dictionary
        if wordOccurences[word] >= minWordOccurence:                    #if it has occurred more than the minimum occurrence input
            words_result.append(word)                                   #add to the result

    return words_result                                                  #return list of words as list as per brief


def task3(task3words, data_reviews, labels):
    wordOccurencesPositive = {}     #dictionary for word occurrences in positive reviews
    wordOccurencesNegative = {}     #dictionary for word occurrences in negative reviews

    for word in task3words: #add words to the dictionaries
        wordOccurencesPositive[word] = 0
        wordOccurencesNegative[word] = 0

    for i, review in enumerate(data_reviews["Review"]):  #loop through each review with enumeration
        words = review.split()                           #split the review into seperate words
        if labels["Sentiment"].values[i] == "positive":  #if the corresponding label is positive
            for word in task3words:                      #for every word in the input set, if the word is in the review add to the word occurrence in positive reviews
                if word in words:
                    wordOccurencesPositive[word] += 1
        else:                                           #if not positive
            for word in task3words:                     #for every word in the input set, if the word is in the review add to the word occurrence in negative reviews
                if word in words:
                    wordOccurencesNegative[word] += 1

        for word in task3words:                         #if not in the review add 0
            if word not in wordOccurencesPositive:
                wordOccurencesPositive[word] = 0

        for word in task3words:                         #if not in the review add 0
            if word not in wordOccurencesNegative:
                wordOccurencesNegative[word] = 0


    print("Words occurring in positive reviews:",wordOccurencesPositive)    #output as per brief
    print("Words occurring in negative reviews:",wordOccurencesNegative)  #output as per brief

    return wordOccurencesPositive, wordOccurencesNegative      #retrun dictionaries


def task4(wordOccurencesPositive, wordOccurencesNegative, training_labels):
    total = len(training_labels)  #total number of reviews
    positive = sum(training_labels.iloc[:, 0] == "positive")  #number of positive reviews
    negative = sum(training_labels.iloc[:, 0] == "negative")  #number of negative reviews


    prior_pos = positive / total  #positive prior
    prior_neg = negative / total  #negative prior
    alpha = 1  #defining alpha as 1

    likelihood_positive = {}  #dictionary for P[word in review |positive review]
    likelihood_negative = {}  #dictionary for P[word in review | negative review]

    for word in wordOccurencesPositive: #loop through word occurrences positive dict and apply laplace smoothing with smoothing factor 1 to get probability for each wor
        likelihood_positive[word] = (wordOccurencesPositive[word] + alpha) / (
                    positive + alpha * len(wordOccurencesPositive))

    for word in wordOccurencesNegative: #loop through word occurrences negative dict and apply laplace smoothing with smoothing factor 1 to get probability for each word
        likelihood_negative[word] = (wordOccurencesNegative[word] + alpha) / (
                    negative + alpha * len(wordOccurencesNegative))

    #print(likelihood_negative)
    #print(likelihood_positive)
    return likelihood_negative, likelihood_positive, prior_pos, prior_neg #return dictionaries and priors as per brief

def task5(review, likelihood_negative, likelihood_positive, prior_pos, prior_neg):
    prediction = [] #array for prediction
    words = review.split() #split review in to single words
    logLikelihood_positive = 0
    logLikelihood_negative = 0

    for word in words: # loop through every word in words
        for key, value in likelihood_positive.items(): # for each key,value pair in the positive dict
            if word == key:
                logLikelihood_positive = logLikelihood_positive + math.log(value) #add the math.log of the value to the corresponding key to the log likelihood

        for key, value in likelihood_negative.items(): # for each key,value pair in the negative dict
            if word == key:
                logLikelihood_negative = logLikelihood_negative + math.log(value)  #add the math.log of the value to the corresponding key to the log likelihood


    if logLikelihood_positive - logLikelihood_negative > math.log(prior_neg) - math.log(prior_pos):  # if the log likeilhood P minues N is greater than the math log of the prior N minusP
        prediction.append(1) # add 1 to the prediction array
        print("Positive")  #output as per brief

    else:
        prediction.append(0) # add 0 to the prediction array
        print("Negative")   #output as per brief

    return prediction


def task6a(new_data_reviews, new_data_labels, wordlength, wordOcc, k):
    skf = StratifiedKFold(n_splits=k) #kfold cross validator
    accuracies = [] #array for accuracies

    for train_index, test_index in skf.split(new_data_reviews, new_data_labels): #get the tarin and test index from the split using the reviews and labels
        new_pred = [] #array for new predicition
        X_train, y_train = new_data_reviews.iloc[train_index, :], new_data_labels.iloc[train_index] #Get the training subset amd corrseponding labels
        X_test, y_test = new_data_reviews.iloc[test_index, :], new_data_labels.iloc[test_index]["Sentiment"].map(
            lambda x: 1 if x == "positive" else 0) #Get the test subset and corrseponding labels mapping to 1 or 0

        word_counts = task2(X_train, wordlength, wordOcc) #get word count from task2
        wordOccPos, wordOccNeg = task3(word_counts, X_train, y_train) # get word occurences from task3

        likeNeg, likePos, priorPos, priorNeg = task4(wordOccPos, wordOccNeg, y_train) #get likelihoods and priors from task 4

        for i in range(len(X_test.index)):  #loop through the length of the test subset from the split
            # print(i)
            review = X_test.iloc[i, :]  #get the review from the subest
            pred = task5(review, likeNeg, likePos, priorPos, priorNeg)  #using the review get a prediction from task 5
            new_pred.append(pred[0])  #add the prediction to the new prediciton array

        accuracy = metrics.accuracy_score(y_test, new_pred)  # get accuracy comparing the new prediciton and actual sentiment
        print("Accuracy:",accuracy)
        accuracies.append(accuracy)   #add to accuracies

    acc_mean = sum(accuracies) / len(accuracies)

    print("Acuuracy mean:" , acc_mean)
    return acc_mean

def task6b(new_data_reviews, new_data_labels, k):

    wordOcc = int(input("Task 6b : Enter minimum word occurrence:"))

    word_lengths = [1,2,3,4,5,6,7,8,9,10]
    best_accuracy = 0
    best_length = 0

    for i in word_lengths:
        result = task6a(new_data_reviews, new_data_labels, i, wordOcc, k)
        if result > best_accuracy:
            best_accuracy = result
            best_length = i

    print("Optimal length: ", best_length)
    print("Accuracy: ", best_accuracy)
    return best_length


def task6c(training_reviews, training_labels, testing_reviews, testing_labels, best_length):
    wordOcc = int(input("Task 6c (Final Evaluation) Enter minimum word occurrence:"))
    new_pred = []
    final_eval = []
    word_counts = task2(training_reviews, best_length, wordOcc)
    wordOccPos, wordOccNeg = task3(word_counts, training_reviews, training_labels)

    likeNeg, likePos, priorPos, priorNeg = task4(wordOccPos, wordOccNeg, training_labels)

    for i in range(len(testing_reviews.index)):
        review = testing_reviews.iloc[i, :]["Review"]
        pred = task5(review, likeNeg, likePos, priorPos, priorNeg)
        new_pred.append(pred[0])

    y_test = testing_labels["Sentiment"].map(lambda x: 1 if x == "positive" else 0)
    confusion = metrics.confusion_matrix(y_test, new_pred)  # create confusion matrix
    print("Confusion matrix:\n", confusion)
    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []

    true_negative.append(confusion[0, 0])  # append index 0,0 to true negative
    true_positive.append(confusion[1, 1])  # append index 1,1 to true positive
    false_negative.append(confusion[1, 0])  # append index 1,0 to false negative
    false_positive.append(confusion[0, 1])  # append index 0,1 to false positive

    print("True positive:", true_positive)  # print the number of true positives
    print("True negative:", true_negative)  # print the number of true negatives
    print("False positive:", false_positive)  # print the number of false positives
    print("False negative:", false_negative)  # print the number of false negatives
    print()

    perc_true_positive = sum(true_positive) / (sum(true_positive) + sum(false_positive))   #% of true positives
    perc_false_positive = 1 - perc_true_positive                            #% of false positives
    perc_true_negative = sum(true_negative) / (sum(true_negative) + sum(false_negative))   #% of true negatives
    perc_false_negative = 1 - perc_true_negative                            #% of false negatives

    class_acc_score = (sum(true_positive) + sum(true_negative)) / (sum(true_positive) + sum(true_negative) + sum(false_positive) + sum(false_negative)) #Classification accuracy score

    #print(class_acc_score)
    final_eval.append(confusion)
    final_eval.append("True positive % : ")
    final_eval.append(perc_true_positive)
    final_eval.append("False positive % : ")
    final_eval.append(perc_false_positive)
    final_eval.append(" True negative % : ")
    final_eval.append(perc_true_negative)
    final_eval.append("False negative % : ")
    final_eval.append(perc_false_negative)
    final_eval.append("Classification accuracy score % : ")
    final_eval.append(class_acc_score)
    final_eval.append("")

    print(final_eval)
    return final_eval



def main():
    training_data_reviews, training_labels, test_data_reviews, test_labels = task1()  #task1

    minWordLength = int(input("Enter minimum word length:"))
    minWordOccurence = int(input("Enter minimum word occurrence:"))

    task3words = task2(training_data_reviews, minWordLength, minWordOccurence)  #task2 returning task3 input
    wordOccurencesPositive, wordOccurencesNegative = task3(task3words, training_data_reviews, training_labels) #task 3 returning task4 input
    likelihood_negative, likelihood_positive, prior_pos, prior_neg= task4(wordOccurencesPositive, wordOccurencesNegative, training_labels)  #task4 returning task 5 input

    review= input("Enter new review:")

    task5(review, likelihood_negative, likelihood_positive, prior_pos, prior_neg) #task5 with new entered review

    k= int(input("Enter how many folds (k):"))  #Input for number of folds

    task6a(training_data_reviews, training_labels, minWordLength, minWordOccurence, k)  #task6
    best_length=task6b(training_data_reviews, training_labels, k)  # Optimal length from task 6b
    task6c(training_data_reviews, training_labels, test_data_reviews, test_labels, best_length) #Final evaluation with optimal length


main()