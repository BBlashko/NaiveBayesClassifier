'''
# Author: Brett A. Blashko
# ID: V00759982
# Purpose:  Computes unsmoothed unigrams and bigrams on the complete works of
#           Shakespeare
'''

from collections import Counter
from tabulate import tabulate

def generateAndPrintTop15(list):
    # Heading format for printing
    info = ["Word", "Count", "Percent"]

    unigrams_top_15 = Counter(list).most_common(15)
    entry_list = []
    for entry in unigrams_top_15:
        entry_list.append([entry[0], str(entry[1]), str(float(entry[1])/len(list)*100)])
    print tabulate(entry_list, headers=info)


# Calling Code
# Get all unigrams (aka words)
with open('shakespeare.txt', 'r') as f:
    unigrams = f.read().split()

# Generate all Bigrams from the Unigrams
bigrams = []
for i, word in enumerate(unigrams):
    if (i + 1 < len(unigrams)):
        bigrams.append(" ".join([word, unigrams[i + 1]]))

#Unigrams
# Count and display number of occurences of unique tokens.
print "\nTop 15 Unigrams:"
generateAndPrintTop15(unigrams)

# bigrams
# Count and display number of occurences of unique tokens.
print "\nTop 15 Bigrams:"
generateAndPrintTop15(bigrams)
