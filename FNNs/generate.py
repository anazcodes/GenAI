import random

import torch


def write_song(model, my_song, vocab, tokens, context_size, device, number_of_words=100):
    """
    Generates text using a trained n-gram language model.

    Given an initial text (`my_song`), the function generates additional words by
    predicting the next word iteratively based on the trained model.

    Parameters:
    model (nn.Module): The trained n-gram language model.
    my_song (str): The initial seed text to start generating words.
    vocab (Vocab): The vocabulary object mapping tokens to indices.
    tokens (list): The preprocessed word tokens used to seed the context window.
    context_size (int): The number of previous words used as context to predict the next word.
    device (torch.device): The device to run inference on.
    number_of_words (int): The number of words to generate (default: 100).

    Returns:
    str: The generated song lyrics as a string.
    """
    # Get the mapping from index to word for decoding predictions
    index_to_token = vocab.get_itos()

    # Loop to generate the desired number of words
    for i in range (number_of_words):
         with torch.no_grad(): # Disable gradient computation for inference

            # Prepare the input context by extracting the last context_size words from tokens
            context = torch.tensor(
                vocab([tokens[i-j-1] for j in range (context_size)])
            ).to(device) # Move to CPU/GPU as required

            # Predict the next word by selecting the word with the highest probability
            word_idx = torch.argmax(model(context)) # Get index of the most likely next word

            # Append the predicted word to the generated text
            my_song += " " + index_to_token[word_idx.detach().item()]

    return my_song # Return generated lyrics


def pickrandomline(song):
    """
    Selects a random line from the given song text.

    This function splits the song into separate lines and randomly picks one of them.

    Parameters:
    song (str): The song lyrics as a multi-line string.

    Returns:
    str: A randomly selected line from the song.
    """

    # Split the song into individual lines
    lines = song.split("\n")

    # Randomly select a line and remove leading/trailing whitespace
    selected_line = random.choice(lines).strip()

    return selected_line # Return the randomly selected line
