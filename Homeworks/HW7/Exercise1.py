#Exercise 1:
import matplotlib.pyplot as plt
from collections import Counter

#Function to read all training emails into lists and count words
def readFiles(file_names):
    words = []
    for file in file_names:
        with open(file) as f:
            words += f.read().split()
    return Counter(words)

normal_email_files = ["train_N_I.txt", "train_N_II.txt", "train_N_III.txt"]
spam_email_files = ["train_S_I.txt", "train_S_II.txt", "train_S_III.txt"]

#Read training emails and split keys and values of counter dictionary for both email types
normal_word_counts = readFiles(normal_email_files)
spam_word_counts = readFiles(spam_email_files)
key_listN = list(normal_word_counts.keys())
val_listN = list(normal_word_counts.values())
key_listS = list(spam_word_counts.keys())
val_listS = list(spam_word_counts.values())

#Read testing emails
with open('testEmail_I.txt') as f:
    test_I = f.read().split()
with open('testEmail_II.txt') as f:
    test_II = f.read().split()
test_emails = [test_I, test_II]

#Perform Naive Bayes Classification with these prior prbabilities
prior_prob_normal = 0.73
prior_prob_spam = 0.27

#Calculate number of words total and number of distinct words in both normal and spam training emails
num_words_normal = sum(val_listN)
num_words_spam = sum(val_listS)
size_normal_list = len(key_listN)
size_spam_list = len(key_listS)

#Calulate probability of each word given normal and spam and account for possible missing words
prob_words_given_normal = []
prob_words_given_spam = []
for i in val_listN:
    prob_words_given_normal.append((i + 1) / (num_words_normal + size_normal_list))
for i in val_listS:
    prob_words_given_spam.append((i + 1) / (num_words_spam + size_spam_list))

#Calculate probability that test email is normal or spam
test_email_probabilities = []
for email in test_emails:
    #Calculate proabbility of test email being normal:
    probability_normal = prior_prob_normal
    for word in email:
        if word in key_listN:
            index = key_listN.index(word)
            prob = prob_words_given_normal[index]
        else:
            prob = 1 / (num_words_normal + size_normal_list)
        probability_normal *= prob

    #Calculate proabbility of test email being spam:
    probability_spam = prior_prob_spam
    for word in email:
        if word in key_listS:
            index = key_listS.index(word)
            prob = prob_words_given_spam[index]
        else:
            prob = 1 / (num_words_spam + size_spam_list)
        probability_spam *= prob
    test_email_probabilities.append([probability_normal, probability_spam])

#Classify based on probabilities
predictions = []
for i in test_email_probabilities:
    if i[0] > i[1]:
        predictions.append('Normal')
    else:
        predictions.append('Spam')

#Print predictions
for i in range(len(predictions)):
    print(f'Test email {i + 1} is predicted to be {predictions[i]}')

#Plot frequency of words in normal and spam emails
#Normal
fig, ax = plt.subplots(1, 2)
ax[0].bar(key_listN, val_listN)
ax[0].set_title('Frequency of Words in Normal Emails')
ax[0].set_xlabel('Word')
ax[0].set_ylabel('Frequency')
ax[0].tick_params(rotation = 90)

#Spam
ax[1].bar(key_listS, val_listS)
ax[1].set_title('Frequency of Words in Spam Emails')
ax[1].set_xlabel('Word')
ax[1].set_ylabel('Frequency')
ax[1].tick_params(rotation = 90)
plt.tight_layout()
plt.show()