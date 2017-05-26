'''
# Author: Brett A. Blashko
# ID: V00759982
# Purpose:  Computes unsmoothed unigrams and bigrams on the complete works of
#           Shakespeare
'''

import sys
import re
from string import punctuation
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
from collections import Counter
from tabulate import tabulate
from random import uniform

def generateAndPrintTop15(list):
    # Heading format for printing
    info = ["Word", "Count", "Percent"]

    counter = Counter(list)
    top_15 = counter.most_common(15)
    entry_list = []
    for entry in top_15:
        entry_list.append([entry[0], str(entry[1]), str(float(entry[1])/len(list)*100)])
    print tabulate(entry_list, headers=info)

def generateNextWord(bigrams, curr_word="<s>"):

    # print "".join(["curr : ", curr_word])

    starting_word_list = [x for x in bigrams if curr_word == x[0].split()[0]]
    # print starting_word_list
    total = sum(x[1] for x in starting_word_list)
    prev = 0.0
    new_prob_list = []
    word = []
    for j, x in enumerate(starting_word_list):
        new_prob_list.append([x[0], prev, prev+x[1]])
        prev += x[1];

    random_percentage = uniform(0, 1) * total
    # print total
    # print " ".join(["random_percentage", str(random_percentage)])
    for x in new_prob_list:
        # print x
        if random_percentage >=x[1]  and random_percentage <= x[2]:
            return x[0].split()[1]
    # print "WORD NOT FOUND"

def generateRandomSentences(num_sentences, bigrams):
    sentences = []
    for i in range(0, num_sentences):
        curr_sentence = []
        word = generateNextWord(bigrams)
        # print word
        curr_sentence.append(word)
        # index = 0
        while (word != "</s>"):
            # if (index > 10):
            #     break
            word = generateNextWord(bigrams, word)
            if (word != "</s>"):
                curr_sentence.append(word)
        final_sentence = re.sub(r'\s([?.!"\';,:](?:\s|$))', r'\1', " ".join(curr_sentence))
        sentences.append(final_sentence)
    return sentences;

# Calling Code
# Get all unigrams (aka words)
with open('shakespeare.txt', 'r') as f:
    sent_tokenize_list = sent_tokenize(f.read())

unigrams_list = []
bigrams_list = []
shannon_bigrams_list = []
word_punct_tokenizer = WordPunctTokenizer()
for i, sentence in enumerate(sent_tokenize_list):
    # generate Unigrams
    unigram_sentence = sentence
    unigrams_list.extend(unigram_sentence.lower().translate(None, punctuation).split())

    # For Bigrams stats
    tokenized_sentence = sentence
    tokenized_sentence = tokenized_sentence.lower().translate(None, punctuation).split()
    for j, word in enumerate(tokenized_sentence):
        if (j + 1 < len(tokenized_sentence)):
            bigrams_list.append(" ".join([word, tokenized_sentence[j + 1]]))

    # For Shannons method
    tokenized_sentence = word_punct_tokenizer.tokenize(sentence)
    # print tokenized_sentence
    for k, word in enumerate(tokenized_sentence):
        if (k == 0):
            shannon_bigrams_list.append(" ".join(["<s>", word]))
            if (k + 1 < len(tokenized_sentence)):
                shannon_bigrams_list.append(" ".join([word, tokenized_sentence[k + 1]]))
        elif(k == len(tokenized_sentence) - 1):
            shannon_bigrams_list.append(" ".join([word, "</s>"]))
        elif (k + 1 < len(tokenized_sentence)):
            shannon_bigrams_list.append(" ".join([word, tokenized_sentence[k + 1]]))


counter_shannon_bigrams_list = Counter(shannon_bigrams_list)
shannon_bigrams_list = [ [k,v, float(v)/len(counter_shannon_bigrams_list)] for k, v in counter_shannon_bigrams_list.items() ]

#Unigrams
# Count and display number of occurences of unique tokens.

print "\nTop 15 Unigrams:"
generateAndPrintTop15(unigrams_list)

# bigrams
# Count and display number of occurences of unique tokens.
print "\nTop 15 Bigrams:"
generateAndPrintTop15(bigrams_list)

sentences = generateRandomSentences(20, shannon_bigrams_list)
print "\nGenerated sentences:"
for sentence in sentences:
    print sentence
