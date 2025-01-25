import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from keras.layers import MaxPooling2D
from keras.preprocessing.image import img_to_array
from pyexpat import model
from tensorflow.keras.layers import Input
# from detector import detect
# from recognizer import recognize
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


class CharacterDetector:
    def __init__(self, train=False, loadFile=""):
        self.res = 180
        self.classes = 26
        self.createModel(self.res,self.classes)
        self.model_name = "eng_alphabets"
        self.word_dict = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
        10: "K",
        11: "L",
        12: "M",
        13: "N",
        14: "O",
        15: "P",
        16: "Q",
        17: "R",
        18: "S",
        19: "T",
        20: "U",
        21: "V",
        22: "W",
        23: "X",
        24: "Y",
        25: "Z",
        }

        if train:
            self.dataset()

        if loadFile:
            if self.loadModel(loadFile):
                print("Model loaded successfully...")
            else:
                import sys

                print("Unable to Load model...")
                sys.exit()

    def loadModel(self, loadFile):
        if self.model:
            self.model.load_weights(loadFile)
            return True
        return False
    
    def dataset(self):
        pass
        
    def createModel(self,res,classes):
        
            self.input_shape = (res, res, 1)
            # Define the input layer
            input_layer = Input(shape=self.input_shape)
            # Create the rest of the model
            x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(input_layer)
            x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
            x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(x)
            x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
            x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="valid")(x)
            x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
            x = Flatten()(x)
            x = Dense(64, activation="relu")(x)
            x = Dense(128, activation="relu")(x)
            output = Dense(classes, activation="softmax", dtype="float32")(x)
            # Create the model
            model = keras.Model(inputs=input_layer, outputs=output)
        
            model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            )
            
            model.summary()
            self.model = model
            
            early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=2,
            verbose=0,
            mode="auto",
        )
        
            self.callbacks = [
            ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.2, 
            patience=1, 
            min_lr=0.0001
            ),
            early_stop,
    #         TensorBoard(log_dir=self.logdir)
            ]
        

    
    def predict(self, img):
        model_name = 'eng_alphabets'
        if type(img) == str:
            img = cv2.imread(img)
        img_copy = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640,480))
        img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
        img_final = cv2.resize(img_thresh, (self.res,self.res))
        img_final = np.reshape(img_final, (1,self.res,self.res,1))
        img_pred = self.word_dict[np.argmax(self.model.predict(img_final))]

        cv2.putText(
                img,
                "Prediction: " + img_pred,
                (20, 410),
                cv2.FONT_HERSHEY_DUPLEX,
                1.3,
                color=(255, 0, 30),
            )
        cv2.imshow("Recognised Character", img)
        if __name__ == "__main__":
            while 1:
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            cv2.destroyAllWindows()
        return img_pred
