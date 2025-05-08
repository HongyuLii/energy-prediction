from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class MLPModel:
    def __init__(self, input_shape, output_shape, n_neurons=128, learning_rate=0.0001, hidden_layers=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers if hidden_layers else [256, 256]  # Default hidden layers
        self.model = None

    def build(self):
        model = Sequential()

        # Input layer
        model.add(Dense(self.n_neurons, input_shape=(self.input_shape,), activation='relu'))

        # Hidden layers
        for neurons in self.hidden_layers:
            model.add(Dense(neurons, activation='relu'))

        # Output layer
        model.add(Dense(self.output_shape, activation='linear'))

        # Compile the model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='log_cosh', metrics=['mae', 'accuracy'])

        self.model = model
    
    def get_model(self):
        return self.model

    def summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            print("Model has not been built yet. Call `build()` first.")

