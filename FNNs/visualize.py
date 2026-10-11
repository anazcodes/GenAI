import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne_embeddings(model, index_to_token, save_path, num_labeled=20):
    """
    Projects a model's word embeddings to 2D with t-SNE and plots them,
    labeling the first `num_labeled` tokens with their word and annotation.

    Parameters:
    model (nn.Module): A trained n-gram language model with an `embeddings` layer.
    index_to_token (list): Mapping from vocabulary index to token string.
    save_path (str): Path (without extension) to save the figure to.
    num_labeled (int): How many of the leading tokens to label (default: 20).
    """

    X = model.embeddings.weight.cpu().detach().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)

    plt.figure()
    labels = []

    for j in range(len(X_2d)):
        if j < num_labeled:
            plt.scatter(X_2d[j, 0], X_2d[j, 1], label=index_to_token[j])
            labels.append(index_to_token[j])
            # Add words as annotations
            plt.annotate(
                index_to_token[j],
                (X_2d[j, 0], X_2d[j, 1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )
        else:
            plt.scatter(X_2d[j, 0], X_2d[j, 1])

    plt.legend(labels, loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_metric_comparison(metrics_list, model_names, ylabel, save_path):
    """
    Plots several per-epoch metric series (e.g. loss or perplexity) on one
    figure, one line per model, and saves the result.

    Parameters:
    metrics_list (list): A list of per-epoch metric sequences, one per model.
    model_names (list): The label for each sequence in `metrics_list`.
    ylabel (str): The metric name used in each line's legend label.
    save_path (str): Path (without extension) to save the figure to.
    """

    plt.figure()
    for metric, model_name in zip(metrics_list, model_names):
        plt.plot(metric, label="{} - {}".format(ylabel, model_name))
        plt.legend()
    plt.savefig(save_path)
    plt.show()
    plt.close()
