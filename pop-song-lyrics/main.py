import warnings
from tqdm import tqdm

warnings.simplefilter('ignore')

import time
from collections import OrderedDict

import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
import time

from sklearn.manifold import TSNE

def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

# Remove all non-word characters (everything except numbers and letters)
def preprocess_string(s):
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", '', s)
    s = re.sub(r"\d", "", s)

    return s

song= """We are no strangers to love
You know the rules and so do I
A full commitments what Im thinking of
You wouldnt get this from any other guy
I just wanna tell you how Im feeling
Gotta make you understand
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Weve known each other for so long
Your hearts been aching but youre too shy to say it
Inside we both know whats been going on
We know the game and were gonna play it
And if you ask me how Im feeling
Dont tell me youre too blind to see
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Weve known each other for so long
Your hearts been aching but youre too shy to say it
Inside we both know whats been going on
We know the game and were gonna play it
I just wanna tell you how Im feeling
Gotta make you understand
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you"""


from nltk.tokenize import word_tokenize

def preprocess(words):
    tokens = word_tokenize(words)
    tokens = [preprocess_string(w) for w in tokens]
    return [w.lower() for w in tokens if len(w) != 0 and (w not in string.punctuation)]

tokens = preprocess(song)
print(f"song tokens: {tokens}")

print(f"first 10 tokens:  {tokens[0:10]}")

vocabulary = set(tokens)
# create a frequency distribution of words
fdist = nltk.FreqDist(tokens)
print(f"Frequency distribution of word 'you': {fdist["you"]}")

plt.bar(list(fdist.keys())[0:10], list(fdist.values())[0:10])
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()
plt.savefig('./pop-song-lyrics/word-freq')

bigrams = nltk.bigrams(tokens)
print(f"bigrams iterator: {bigrams}")

# convert a generator into a list, where each element of the list is a bigram
my_bigrams = list(nltk.bigrams(tokens))

print(f"sample bigrams: {my_bigrams}")

freq_bigrams = nltk.FreqDist(nltk.bigrams(tokens))
print(f"freq_bigrams: {freq_bigrams[("gon", "na")]}")

for mb in my_bigrams[0:10]:
    print(f"biagram: {mb} freq:{freq_bigrams[mb]}")

word = "strangers"
vocab_probabilities = {}
for nw in vocabulary:
    vocab_probabilities [nw] = freq_bigrams[(word, nw)]/fdist[word]

vocab_probabilities = sorted(vocab_probabilities.items(), key = lambda x:x[1], reverse=True)

print(f"vocab probabilities: {vocab_probabilities[0:4]}")

def make_predictions(my_words, freq_grams, normalize = 1, vocabulary= vocabulary):
    """
    Generate predictions for the conditional probability of the next word given a sequence.

    Args:
        my_words (list): A list of words in the input sequence.
        freq_grams (dict): A dictionary containing frequency of n-grams.
        normlize (int): A normalization factor for calculating probabilities.
        vocabulary (list): A list of words in the vocabulary.

    Returns:
        list: A list of predicted words along with their probabilities, sorted in descending order.
    """


    vocab_probabilities = {}  # Initialize a dictionary to store predicted word probabilities
    context_size = len(list(freq_grams.keys())[0]) # Determine the context size from n-grams keys
    print(f"context size: {context_size-1}")
    # Preprocess input words and take only the relevant context words
    my_tokens = preprocess(my_words)[0:context_size-1]

    # Calculate probabilities for each word in the vocabulary given the context
    for nw in vocabulary:
        temp = my_tokens.copy()
        temp.append(nw) # Add the next word to the context

        print(f"Temp: {temp} Normalize: {normalize}")

        # Calculate the conditional probability using the frequency information
        if normalize != 0:
            vocab_probabilities[nw] = freq_grams[tuple(temp)]/ normalize
        else:
            vocab_probabilities[nw] = freq_grams[tuple(temp)]


        print(f"Frequency: {freq_grams[tuple(temp)]}")

    # Sort the predicted words based on their probabilities in descending order
    vocab_probabilities = sorted(vocab_probabilities.items(), key = lambda x:x[1], reverse=True)

    return vocab_probabilities # Return the sorted list of predicted words and their probabilities


my_words = "are"
vocab_probabilities = make_predictions(my_words, freq_bigrams, normalize=fdist["are"])
# print(f"vocab probabilities: {vocab_probabilities[0:10]}")

my_song =""
for w in tokens[0:100]:
    my_word = make_predictions(w, freq_bigrams)[0][0]
    my_song += " " + my_word




my_song = "i"

for i in range (100):
    my_word = make_predictions(my_word, freq_bigrams)[0][0]
    my_song += " " + my_word


print(f"\n\nBigram Lyrics (context size = 1):\n\n{my_song}\n")



freq_trigrams = nltk.FreqDist(nltk.trigrams(tokens))
# print(f"freq_trigrams: {freq_trigrams}")

make_predictions ("so do", freq_trigrams, normalize=freq_bigrams[("do", "i")])[0:10]


my_song = ""

w1 = tokens[0]
for w2 in tokens[0:100]:
    gram= w1 +' '+w2
    my_word = make_predictions(gram, freq_trigrams)[0][0]
    my_song += " "+ my_word
    w1 = w2

print(f"\n\nTrigram Lyrics (context size = 2): \n\n{my_song}\n")
