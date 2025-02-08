# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, AlphaDropout , LayerNormalization , BatchNormalization , Dropout
from tensorflow_addons.layers import WeightNormalization

# %%
data = pd.read_csv("HTRU_2.csv") # Reading the Data and finding the shape
X =  data.iloc[:,:8]
y =  data.iloc[:,-1:]
X.shape, y.shape

# %%
#Split the Dataset and convert it into numpy
X_train , X_test , y_train , y_test =train_test_split(X,y,test_size=0.2)
temp_data, X_test, temp_labels, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(temp_data, temp_labels, test_size=0.25, random_state=42)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
X_val = X_val.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_val = y_val.to_numpy()

# %%
import keras
from keras.models import Sequential
from keras.layers import Dense

class SelfNeuralNetwork:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(512, activation='selu', kernel_initializer='lecun_normal', bias_initializer='zeros', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.05))

        for _ in range(self.num_layers - 1):
            model.add(Dense(256, activation='selu', kernel_initializer='lecun_normal', bias_initializer='zeros'))
            model.add(Dropout(0.05))

        model.add(Dense(2, activation='softmax', kernel_initializer='lecun_normal', bias_initializer='zeros'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])
        return model

    def train_and_evaluate(self, X_train, y_train, X_test, y_test , batch_size, epochs):
        # Train the model
        self.model.fit(X_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=(X_test, y_test))

        # Evaluate the model
        score = self.model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score


# %%
num_layer_list=[2,3,4,6,8,16,32]
epochs=[10,50,100,150]
batch_sizes=[64,128,256]
learning_rate =[32,64,128]
# Iterate over each target column
import matplotlib.pyplot as plt
for num_layers in num_layer_list:
    targets=[]
    snn_accuracy=[]
    snn_loss=[]
    layer_accuracy=[]
    layer_loss=[]
    weight_accuracy=[]
    weight_loss=[]


    mlp_classifier=SelfNeuralNetwork(num_layers)
    score = mlp_classifier.train_and_evaluate(X_train, y_train, X_test, y_test)
    snn_accuracy.append(score[1])
    snn_loss.append(score[0])



