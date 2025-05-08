from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class ModelTrainer():
    
    def __init__(self, model, features_training_data, target_training_data, features_eval_data, target_eval_data):
        self.model = model
        self.history = None
        self.early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        self.x_train = features_training_data
        self.y_train = target_training_data
        self.x_eval = features_eval_data
        self.y_eval = target_eval_data
    
    def train(self, epochs=40, batch_size=32):
        if self.x_eval is None or self.y_eval is None:  # If no eval data is provided
            self.history = self.model.fit(
                self.x_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=2
            )
        else:
            self.history = self.model.fit(
                self.x_train, self.y_train,
                validation_data=(self.x_eval, self.y_eval),
                epochs=epochs,
                batch_size=batch_size,
                verbose=2
            )
    
    def get_training_history(self):
        return self.history
    
    def model_evaluate(self, x_test, y_test):
        # Evaluate the model
        accuray, loss, mae = self.model.evaluate(x_test, y_test)
        return accuray, loss, mae
    
    def custom_evaluate(self, predictions, actual):
        errors = predictions - actual
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        print('MAE: {:.2f}'.format(mae))
        print('RMSE: {:.2f}'.format(rmse))
        print('')
        print('')
        return mae, rmse
