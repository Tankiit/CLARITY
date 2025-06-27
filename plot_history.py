import json
import matplotlib.pyplot as plt
import os

# Paths to history files
ag_news_history_path = os.path.join('results', 'distilbert-base-uncased_ag_news_1', 'ag_news_history.json')
sst2_history_path = os.path.join('results', 'distilbert-base-uncased_sst2_1', 'sst2_history.json')

def load_history(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_history(history, title_prefix, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 4))
    # Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()
    plt.grid(True)
    # Validation F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_f1"], label="Val F1", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title(f"{title_prefix} Validation F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_history.png"))
    plt.close()

def plot_train_val_loss(history, title_prefix, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_train_val_loss.png"))
    plt.close()

def main():
    # AG News
    ag_news_history = load_history(ag_news_history_path)
    plot_history(ag_news_history, 'AG News')
    plot_train_val_loss(ag_news_history, 'AG News')

    # SST2
    sst2_history = load_history(sst2_history_path)
    plot_history(sst2_history, 'SST2')
    plot_train_val_loss(sst2_history, 'SST2')

    print("Plots saved to the 'plots' directory.")

    # Yelp: No history/metrics file found, only config is available
    # If you want to visualize config/hyperparameters, add code here
    # Example (uncomment to use):
    # yelp_config_path = os.path.join('results', 'optimized_yelp', 'optimized_yelp', 'yelp_polarity_config.json')
    # with open(yelp_config_path, 'r') as f:
    #     yelp_config = json.load(f)
    #     print(json.dumps(yelp_config, indent=2))

if __name__ == '__main__':
    main()