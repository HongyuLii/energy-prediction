import torch
import numpy as np

class ModelTrainer:
    def __init__(self, model, features_training_data, target_training_data,
                 features_eval_data=None, target_eval_data=None,
                 device='cpu', loss_fn=None):

        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn or torch.nn.MSELoss()

        self.x_train = torch.tensor(features_training_data, dtype=torch.float32).to(device)
        self.y_train = torch.tensor(target_training_data, dtype=torch.float32).to(device)

        if features_eval_data is not None:
            self.x_eval = torch.tensor(features_eval_data, dtype=torch.float32).to(device)
            self.y_eval = torch.tensor(target_eval_data, dtype=torch.float32).to(device)
        else:
            self.x_eval, self.y_eval = None, None

        self.history = {'train_loss': [], 'eval_loss': []}

    def train(self, epochs=40, batch_size=32, patience=10, learning_rate=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        #loss_fn = torch.nn.MSELoss()
        loss_fn = self.loss_fn

        best_eval_loss = float('inf')
        patience_counter = 0

        N = self.x_train.shape[0]

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            permutation = torch.randperm(N)
            for i in range(0, N, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = self.x_train[indices], self.y_train[indices]

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= (N // batch_size)
            self.history['train_loss'].append(train_loss)

            eval_loss = None
            if self.x_eval is not None:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(self.x_eval)
                    eval_loss = loss_fn(outputs, self.y_eval).item()
                self.history['eval_loss'].append(eval_loss)

                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Eval Loss = {eval_loss:.4f}")

                # Early stopping
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    self.model.load_state_dict(best_model_state)
                    break
            else:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

    def predict(self, x):
        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(x_tensor)
        return predictions.cpu().numpy()

    def custom_evaluate(self, predictions, actual):
        errors = predictions - actual
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        print('MAE: {:.4f}'.format(mae))
        print('RMSE: {:.4f}'.format(rmse))
        print('')
        return mae, rmse

    def get_training_history(self):
        return self.history
