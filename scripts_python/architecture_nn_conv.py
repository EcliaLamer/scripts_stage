import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, Conv2D
from tensorflow.keras.layers import MaxPooling3D, MaxPooling2D
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt
import glob
from os.path import basename, splitext

# Ce script permet de selectionner la meilleure architecture en affichant sur le prompt
# les valeurs de test de la loss et de l'accuracy de chaque archi.

# On importe les datas brutes
# on récup les données
name_data_list = list(glob.glob('output/example_first_K15/data*.npy'))
name_target_list = list(glob.glob('output/example_first_K15/start*.npy'))


# On crée une fct pour ordonner les données et les targets
def get_nb_process_and_simu_from_path(path_data):
    """
    Les donnees ont ete cree par de multiples process independant. Cette fonction
    extrait l'identifiant du process ayant cree le fichier de donnee a partir du
    nom du fichier.

    Cette fonction est utilise pour matcher les donnee avec leur target.
    """
    name_data = basename(path_data) # recup le nom de l'image
    name_data_without_extension = name_data.split('.')[0]
    name_process_with_process = name_data_without_extension.split('_')[1]
    name_process = name_process_with_process.replace("process","")
    nb_process = int(name_process)
    return nb_process

# on l'applique
name_data_list = sorted(name_data_list, key=get_nb_process_and_simu_from_path)
name_target_list = sorted(name_target_list, key=get_nb_process_and_simu_from_path)

# on transforme et on normalise les donnees\
list_datas = []
for filename in name_data_list:
    data = 0.02 * np.asfarray(np.load(filename))
    list_datas.append(data)
array_data_set_tot = np.concatenate((list_datas),axis=0)

## pour les targets
list_starts = []
for filename in name_target_list:
    start_float = 0.01 * np.asfarray(np.load(filename))
    list_starts.append(start_float)
array_targets = np.concatenate((list_starts),axis=0)


# -----------------------------------------------------------------
# preparation du NN
# On rajoute une dim pour grayscale
array_data_set = np.expand_dims(array_data_set_tot, axis=-1)

# On split en deux ensemble : train et test (par indexing)
indexes = np.arange(np.shape(array_data_set)[0])
np.random.seed(46)
np.random.shuffle(indexes)

train_index = indexes[: int(0.8 * np.shape(array_data_set)[0])]
test_index = indexes[int(0.8 * np.shape(array_data_set)[0]) :]

train_dataset = array_data_set[train_index]
test_dataset = array_data_set[test_index]

train_target = array_targets[train_index]
test_target = array_targets[test_index]


# Define modifiable training hyperparameters.
epochs = 10
batch_size = 64

# Reseaux de neurones

# -----------------------------------------------------------------
# test premiere architecture
# Initialisation du modèle
model_0 = Sequential()

# Première couche convolutive
model_0.add(Conv2D(64, (2, 2), padding='same', input_shape = (100, 100, 1), activation = 'relu'))
model_0.add(Conv2D(64, (2, 2), activation='relu'))
model_0.add(MaxPooling2D(pool_size=(2, 2)))
model_0.add(Dropout(0.2))

# Ajout d'une deuxième couche convolutive
model_0.add(Conv2D(32, (3, 3), padding='same', activation = 'relu'))
model_0.add(Conv2D(32, (3, 3), activation='relu'))
model_0.add(MaxPooling2D(pool_size=(2, 2)))
model_0.add(Dropout(0.2))

# Flatten 
model_0.add(Flatten())

# Connexion complète
model_0.add(Dense(units = 32, activation = 'relu'))
model_0.add(Dense(units = 16, activation = 'relu'))
model_0.add(Dense(units = 2, activation = 'relu'))


# Compile
model_0.compile(
    loss='mean_squared_error', optimizer='adam',metrics=['accuracy']
)

model_0.summary()

# Fit the model to the training data.
model_0.fit(
    train_dataset,
    train_target,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    verbose=2
)

score_0 = model_0.evaluate(test_dataset, test_target, verbose=2)
print('score model 0 :', score_0)


# -----------------------------------------------------------------
# test deuxieme architecture
# Initialisation du modèle
model_1 = Sequential()

# Première couche convolutive
model_1.add(Conv2D(64, (2, 2), padding='same', input_shape = (100, 100, 1), activation = 'relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Conv2D(64, (2, 2), activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Dropout(0.2))

# Ajout d'une deuxième couche convolutive
model_1.add(Conv2D(32, (3, 3), padding='same', activation = 'relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Conv2D(32, (3, 3), activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Dropout(0.2))

# Flatten 
model_1.add(Flatten())

# Connexion complète
model_1.add(Dense(units = 32, activation = 'relu'))
model_1.add(Dense(units = 16, activation = 'relu'))
model_1.add(Dense(units = 2, activation = 'relu'))


# Compile
model_1.compile(
    loss='mean_squared_error', optimizer='adam',metrics=['accuracy']
)

model_1.summary()


# Fit the model to the training data.
model_1.fit(
    train_dataset,
    train_target,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    verbose=2
)

score_1 = model_1.evaluate(test_dataset, test_target, verbose=2)
print('score model 1 :', score_1)


# -----------------------------------------------------------------
# test troisieme architecture
# Initialisation du modèle 2 : moins de dense
model_2 = Sequential()

# Première couche convolutive
model_2.add(Conv2D(64, (2, 2), padding='same', input_shape = (100, 100, 1), activation = 'relu'))
model_2.add(Conv2D(64, (2, 2), activation='relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
model_2.add(Dropout(0.2))

# Ajout d'une deuxième couche convolutive
model_2.add(Conv2D(32, (3, 3), padding='same', activation = 'relu'))
model_2.add(Conv2D(32, (3, 3), activation='relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
model_2.add(Dropout(0.2))

# Flatten 
model_2.add(Flatten())

# Connexion complète
model_2.add(Dense(units = 32, activation = 'relu'))
model_2.add(Dense(units = 2, activation = 'relu'))


# Compile
model_2.compile(
    loss='mean_squared_error', optimizer='adam',metrics=['accuracy']
)

model_2.summary()

# Fit the model to the training data.
model_2.fit(
    train_dataset,
    train_target,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    verbose=2
)

score_2 = model_2.evaluate(test_dataset, test_target, verbose=2)
print('score model 2 :', score_2)


# -----------------------------------------------------------------
# test quatrieme architecture
# Initialisation du modèle 3
model_3 = Sequential()

# Première couche convolutive
model_3.add(Conv2D(64, (2, 2), padding='same', input_shape = (100, 100, 1), activation = 'relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2)))
model_3.add(Conv2D(64, (2, 2), activation='relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2)))
model_3.add(Dropout(0.2))

# Ajout d'une deuxième couche convolutive
model_3.add(Conv2D(32, (3, 3), padding='same', activation = 'relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2)))
model_3.add(Conv2D(32, (3, 3), activation='relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2)))
model_3.add(Dropout(0.2))

# Flatten 
model_3.add(Flatten())

# Connexion complète
model_3.add(Dense(units = 32, activation = 'relu'))
model_3.add(Dense(units = 2, activation = 'relu'))


# Compile
model_3.compile(
    loss='mean_squared_error', optimizer='adam',metrics=['accuracy']
)

model_3.summary()


# Fit the model to the training data.
model_3.fit(
    train_dataset,
    train_target,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    verbose=2
)

score_3 = model_3.evaluate(test_dataset, test_target, verbose=2)
print('score model 3 :', score_3)