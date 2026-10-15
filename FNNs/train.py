import sys
from tqdm import tqdm

from generate import pickrandomline, write_song


def train(dataloader, model, song, optimizer, criterion, vocab, tokens, context_size,
          device, number_of_epochs=100, show=10):
    """
    Args:
        dataloader (DataLoader): DataLoader containing training data.
        model (nn.Module): Neural network model to be trained.
        song (str): The full song lyrics, used to seed progress-check generations.
        optimizer (Optimizer): Optimizer used to update the model's parameters.
        criterion (Loss): Loss function used to score predictions against targets.
        vocab (Vocab): The vocabulary object mapping tokens to indices.
        tokens (list): The preprocessed word tokens for the song.
        context_size (int): The number of previous words used as context.
        device (torch.device): The device to run training on.
        number_of_epochs (int, optional): Number of epochs for training. Default is 100.
        show (int, optional): Interval for displaying progress. Default is 10.

    Returns:
        list: List containing loss values for each epoch.
    """


    MY_LOSS = [] # List to store loss values for each epoch


    # iterate over the specified number of epochs
    for epoch in tqdm(range(number_of_epochs)):
        total_loss = 0 # Initialize total loss for the current epoch
        my_song ="" # Intialize a string to store the generated song

        # Iterate over batches in the dataloader
        for context, target in dataloader:
            model.zero_grad() # Sero the gradients to avoid accumulation
            predicted = model(context) # Forward pass through the model to get predictions
           
            loss = criterion(predicted, target.reshape(-1)) # Calculate the loss
            total_loss += loss.item() # Accumulate the loss

            # print(f"context: {context} predicted: {predicted} target:{target} target.reshape: {target.reshape(-1)} loss: {loss}")
            # sys.exit(0)           

            loss.backward() # Backpropagation to compute gradients
            optimizer.step() # Update model parameters using the optimizer

        # Display progrss and generate song at specified intervals
        if epoch % show == 0:
            selected_line = pickrandomline(song)
            my_song += write_song(model, selected_line, vocab, tokens, context_size, device) # Generate song using the model

            print("Generated song: \n", my_song)

        MY_LOSS.append(total_loss/len(dataloader)) # Append the total loss for the epoch to MY_LOSS list

    return MY_LOSS # Return the list of mean loss values for each epoch
