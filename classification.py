import math
import glob
from ngram import NGram

#Train positive Unigram Model
model_positive = NGram(ngram_size = 1, vocab_file='imdb.vocab')
model_positive.train('./sentiment_data/train/pos')

#Train negative Unigram Model
model_negative = NGram(ngram_size = 1, vocab_file='imdb.vocab')
model_negative.train('./sentiment_data/train/neg')

#Load Test Data into Arrays and Get Data Counts (Negative)
data_neg = []
neg_file_count = 0
for file in glob.glob("./sentiment_data/test/neg/*.txt"):
    with open(file, 'r') as f:
        data_neg.append(f.read())
        neg_file_count+=1

#Load Test Data into Arrays and Get Data Counts (Positive)
data_pos = []
pos_file_count = 0
for file in glob.glob("./sentiment_data/test/pos/*.txt"):
    with open(file, 'r') as f:
        data_pos.append(f.read())
        pos_file_count+=1


#Calculate prior probabilities (log)
prior_prob_n = math.log(neg_file_count/(neg_file_count+pos_file_count))
prior_prob_p = math.log(pos_file_count/(neg_file_count+pos_file_count)) #Or log(1-prior_prob_n) but hey...

#Initialize Variables
true_positive_pos = 0
false_positive_pos = 0
true_negative_pos = 0
false_negative_pos = 0
true_positive_neg = 0
false_positive_neg = 0
true_negative_neg = 0
false_negative_neg = 0

#Calculate Log Likelihoods for Positive Test Data
for i in range(len(data_pos)):
    MLE_counts_pos = {}
    likelihood_pos = 0
    likelihood_neg = 0
    file = data_pos[i]
    #Get MLE counts
    MLE_counts_pos = model_positive.mle(data=file, ngrams={})
    #Calculate log likelihood for each Model
    for ngram in MLE_counts_pos['1gram']:
        likelihood_neg += math.log(model_negative.prob((ngram,), params = {'k': 0.1}))
        likelihood_pos += math.log(model_positive.prob((ngram,), params = {'k': 0.1}))
    #Add prior probability to log likelihood
    total_likelihood_neg = likelihood_neg + prior_prob_n
    total_likelihood_pos = likelihood_pos + prior_prob_p
    #Compare log likelihoods
    if total_likelihood_pos > total_likelihood_neg:
        true_positive_pos += 1
        true_negative_neg += 1
    else:
        false_positive_neg += 1
        false_negative_pos += 1

#Calculate Log Likelihoods for Negative Test Data
for i in range(len(data_neg)):
    MLE_counts_neg = {}
    likelihood_pos = 0
    likelihood_neg = 0
    file = data_neg[i]
    #Get MLE counts for each file
    MLE_counts_neg = model_negative.mle(data=file, ngrams={})
    #Calculate log likelihood for each Model
    for ngram in MLE_counts_neg['1gram']:
        likelihood_neg += math.log(model_negative.prob((ngram,), params = {'k': 0.1}))
        likelihood_pos += math.log(model_positive.prob((ngram,), params = {'k': 0.1}))
    #Add prior probability to log likelihood
    total_likelihood_neg = likelihood_neg + prior_prob_n
    total_likelihood_pos = likelihood_pos + prior_prob_p
    #Compare log likelihoods
    if total_likelihood_neg > total_likelihood_pos:
        true_positive_neg += 1
        true_negative_pos += 1
    else:
        false_positive_pos += 1
        false_negative_neg += 1
#Calculate Accuracy
accuracy = (true_positive_pos + true_negative_pos)/(true_positive_pos + true_negative_pos + false_positive_pos + false_negative_pos)
print("Accuracy: ", accuracy)
#Calculate Precision for Positive Model
precision_pos = true_positive_pos/(true_positive_pos + false_positive_pos)
print("Positive")
print("\tPrecision: ", precision_pos)
#Calculate Recall for Positive Model
recall_pos = true_positive_pos/(true_positive_pos + false_negative_pos)
print("\tRecall: ", recall_pos)
#Calculate F1 Score for Positive Model
f1_pos = 2 * ((precision_pos * recall_pos)/(precision_pos + recall_pos))
print("\tF1 Score: ", f1_pos)
#Calculate Precision for Negative Model
precision_neg = true_positive_neg/(true_positive_neg + false_positive_neg)
print("Negative")
print("\tPrecision: ", precision_neg)
#Calculate Recall for Negative Model
recall_neg = true_positive_neg/(true_positive_neg + false_negative_neg)
print("\tRecall: ", recall_neg)
#Calculate F1 Score for Negative Model
f1_neg = 2 * ((precision_neg * recall_neg)/(precision_neg + recall_neg))
print("\tF1 Score: ", f1_neg)


