import pandas as pd
import matplotlib.pyplot as plt


class DataVisualizer:
    def __init__(self, save_path = ''):
        self.save_path = save_path

    def filter_and_align_data(self, actual, prediction, scale=5000):
        # Filter data to focus on prices less than the scale value
        actual_filtered = actual[actual.iloc[:, 0] < scale]
        pred_filtered = prediction[prediction.iloc[:, 0] < scale]

        # Ensure alignment of indices after filtering
        aligned_actual, aligned_pred = actual_filtered.align(pred_filtered, join='inner', axis=0)
        return aligned_actual, aligned_pred

    def plot_prediction_comparison(self, actual = None, prediction = None, scale=5000, title='Actual vs Predicted SCED Prices', y_label='SCED Price', save_path=None):
        aligned_actual, aligned_pred = self.filter_and_align_data(actual, prediction, scale)

        # Plot the filtered datasets
        plt.figure(figsize=(12, 6))
        plt.plot(aligned_actual.index, aligned_actual.iloc[:, 0], label='Actual SCED', alpha=0.8, linestyle='-', marker='o')
        plt.plot(aligned_pred.index, aligned_pred.iloc[:, 0], label='Predicted SCED', alpha=0.8, linestyle='--', marker='x')

        # Adding labels, title, and legend
        plt.title(title, fontsize=16)
        plt.xlabel('Timestamp', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.5)
        plt.tight_layout()

        # Save the plot if a path is specified
        if save_path:
            plt.savefig(save_path)

        # Show the plot
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training and validation loss."""
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss', linestyle='--')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

