import re
import string

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from nltk.tokenize import word_tokenize


tokenizer = get_tokenizer("basic_english")


def preprocess_string(s):
    """
    Preprocesses a given string by performing the following steps:

    1. Removes all non-word characters (excluding letters and numbers).
    2. Removes all whitespace characters.
    3. Removes all numeric digits.

    Parameters:
    s (str): The input string to be cleaned.

    Returns:
    str: The processed string with only alphabetic characters, no spaces, and no digits.
    """

    # Remove all non-word characters (everything except letters and numbers)
    # \w matches any word character (letters, numbers, and underscores)
    # \s matches any whitespace characters
    # ^ inside [] negates the selection, so [^\w\s] matches anything that's NOT a word character or whitespace.
    s = re.sub(r"[^\w\s]", '', s)

    # Remove all whitespace characters (spaces, tabs, newlines)
    # \s+ matches one or more whitespace characters.
    s = re.sub(r"\s+", '', s)

    # Remove all digits (0-9)
    # \d matches any digit character.
    s = re.sub(r"\d", '', s)

    return s


# NLTK pipeline
def preprocess(words):
    """
    Preprocesses a given text by tokenizing it, cleaning individual words, and
    converting them to lowercase while removing empty or punctuation tokens.

    Steps:
    1. Tokenization: Splits the input text into individual word tokens.
    2. Cleaning: Applies `preprocess_string()` to remove non-word characters,
       spaces, and digits from each token.
    3. Normalization: Converts all tokens to lowercase.
    4. Filtering: Removes empty strings and punctuation tokens.

    Parameters:
    words (str): The input text to be tokenized and preprocessed.

    Returns:
    list: A list of cleaned, lowercase tokens.
    """

    # Tokenize the input text into words
    tokens = word_tokenize(words)
    print(f"word tokenize output: {tokens}")

    # Apply preprocessing to each token (removes unwanted characters)
    tokens = [preprocess_string(w) for w in tokens]

    # Convert tokens to lowercase and remove empty strings or punctuation
    return [w.lower() for w in tokens if len(w) != 0 and w not in string.punctuation]


def tokenizetext(tokens):
    """
    Builds a vocabulary from an already-tokenized list of words.

    Building the vocabulary from this same `tokens` list (rather than
    re-tokenizing the raw song text with a different tokenizer) guarantees
    every word used elsewhere in the pipeline is actually present in the
    vocabulary, instead of silently falling back to "<unk>" whenever the
    two tokenizers disagree on word boundaries.

    Steps:
    1. Vocabulary Building: Constructs a vocabulary from the tokens,
       including a special "<unk>" token to handle out-of-vocabulary words.
    2. Default Indexing: Sets the default index for unknown words, ensuring
       that any unseen tokens are mapped to "<unk>".

    Parameters:
    tokens (list): The preprocessed word tokens (e.g. from `preprocess()`).

    Returns:
    vocab (Vocab): A vocabulary object mapping tokens to their corresponding indices.
    """

    # Wrap each token as its own single-token "sentence", since
    # build_vocab_from_iterator expects an iterable of token lists.
    tokenized_song = ([token] for token in tokens)

    # Build vocabulary from tokenized text
    # The function 'build_vocab_from_iterator' constructs a vocabulary by iterating
    # over the tokenized words. The special token "<unk>" is added to handle words that are not present in the vocabulary.
    vocab = build_vocab_from_iterator(tokenized_song, specials = ["<unk>"])

    # Set the default index for unknown words
    # The default index is set to the index of "<unk>" so that any word not found in the
    # vocabulary is mapped to this token, preventing errors during lookup.UnboundLocalError

    vocab.set_default_index(vocab["<unk>"])

    return vocab


def genngrams(tokens, context_size):
    """
    Generates n-grams from a list of tokens, where each n-gram consists of a
    context (previous words) and a target (next word).

    The function constructs a list of tuples where:
    - The first element is a list of `context_size` previous words.
    - The second element is the target word that follows the context.

    Parameters:
    tokens (list): A list of preprocessed word tokens.
    context_size (int): The number of previous words used as context.

    Returns:
    list: A list of tuples representing n-grams.
          Each tuple contains (context_words, target_word).
    """

    # Generate n-grams
    # Iterate through the tokens starting from index context_size to the end
    # For each token at position 'i', extract thee previous context_size words as context
    ngrams = [
        (
            [tokens[i-j-1] for j in range (context_size)], # Context words (previous words)
            tokens[i] # Target word (the word to predict)


        )

        for i in range (context_size, len(tokens))
    ]

    return ngrams


def make_collate_fn(vocab, context_size, device):
    """
    Builds a `collate_fn` for a DataLoader that converts a batch of tokens into
    (context, target) index tensors, using the given vocab/context_size/device.

    Parameters:
    vocab (Vocab): The vocabulary object used to map tokens to indices.
    context_size (int): The number of previous words used as context.
    device (torch.device): The device to move the resulting tensors to.

    Returns:
    callable: A `collate_batch(batch)` function suitable for `DataLoader(collate_fn=...)`.
    """

    def collate_batch(batch):
        """
        Processes a batch of text data into input (context) and output (target) tensors
        for training a language model.

        The function extracts:
        - `context`: A list of word indices representing the context words for each target word.
        - `target`: A list of word indices representing the target word to predict.

        Parameters:
        batch (list): A list of tokenized words (strings).

        Returns:
        tuple: Two PyTorch tensors: (context_tensor, target_tensor)
               - context_tensor: Tensor of shape (batch_size - context_size, context_size),
                 containing the word indices of context words.
               - target_tensor: Tensor of shape (batch_size - context_size,),
                 containing the word indices of target words.
        """

        batch_size = len(batch)  # Get the size of the batch
        context, target = [], [] # Initialize lists for context and target words

        # Loop through the batch, ensuring enough previous words exist for context
        for i in range(context_size, batch_size):
            # Convert the target word to its index using the vocabulary
            target.append(vocab([batch[i]]))

            # Convert the previous context_size words to indices using the vocabulary
            context.append(vocab([batch[i-j-1] for j in range(context_size)]))

        # Convert lists to PyTorch tensors and move them to the appropriate device (CPU/GPU)
        return torch.tensor(context).to(device), torch.tensor(target).to(device).reshape(-1)

    return collate_batch
