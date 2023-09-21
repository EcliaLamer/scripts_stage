from sampy.data_processing.csv_manager import CsvManager # pour gerer les hyper-param
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
from numpy import savez_compressed
from tensorflow.keras import layers, losses, optimizers
#from tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing import sequence
#from tensorflow.keras.utils import to_categorical
#import matplotlib.pyplot as plt
import glob
from os.path import basename, splitext

# ICI nous avons choisi une architecture du NN. On teste alors des hyper-parametre (conv, dense et dropout)

# On importe les datas brutes
# on recup lelien des donnees de calcul canada
name_data_list = list(glob.glob('output/example_first_K15/data*.npy'))
name_target_list = list(glob.glob('output/example_first_K15/start*.npy'))

# On cree une fct pour ordonner les donnees et les target
def get_nb_process_and_simu_from_path(path_data):
    name_data = basename(path_data) # recup le nom de l'image
    name_data_without_extension = name_data.split('.')[0]
    name_process_with_process = name_data_without_extension.split('_')[1]
    name_process = name_process_with_process.replace("process","")
    nb_process = int(name_process)
    return nb_process

# on l'applique
name_data_list = sorted(name_data_list, key=get_nb_process_and_simu_from_path)
name_target_list = sorted(name_target_list, key=get_nb_process_and_simu_from_path)

# on les transforme et on les normalise
## pour les datas

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

# preparation NN

# On rajoute une dim pour grayscale
array_data_set = np.expand_dims(array_data_set_tot, axis=-1)
print(np.shape(array_data_set))

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
np.savez_compressed('output/select_hyperparam/test_target', a= str(test_target))

# Inspect the dataset
print("Training Dataset Shapes: " + str(train_dataset.shape) + ", " + str(train_target.shape))
print("Testing Dataset Shapes: " + str(test_dataset.shape) + ", " + str(test_target.shape))

# Gestion des hyperparametres
dict_types = {'conv_layer_1': int,
              'conv_layer_2': int,
              'dense_layer_1': int,
              'dense_layer_2': int,
              'drop_out_1': float,
              'id_simu': str}

csv_manager = CsvManager('input/csv_hyper_param_nn_conv.csv', ';', dict_types=dict_types)

param = csv_manager.get_parameters()

# reseau de neurones
while param is not None:
    # Initialisation du modèle
    model = Sequential()

    # Première couche convolutive
    model.add(Conv2D(param.conv_layer_1, (2, 2), padding='same', input_shape = (100, 100, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(param.conv_layer_1, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(param.drop_out_1))

    # Ajout d'une deuxième couche convolutive
    model.add(Conv2D(param.conv_layer_2, (3, 3), padding='same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(param.conv_layer_2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(param.drop_out_1))

    # Flatten 
    model.add(Flatten())

    # Connexion complète
    model.add(Dense(units = param.dense_layer_1, activation = 'relu'))
    model.add(Dense(units = param.dense_layer_2, activation = 'relu'))
    model.add(Dense(units = 2, activation = 'relu'))

    # Compile
    model.compile(
        loss='mean_squared_error', optimizer='adam', metrics=['accuracy']
    )

    model.summary()

    # Define modifiable training hyperparameters.
    epochs = 2
    batch_size = 5

    # Fit the model to the training data.
    hist = model.fit(
        train_dataset,
        train_target,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=2)

    # Enregistrement des scores 

    # accuracy entrainement
    train_acc = hist.history['acc']
    np.savez_compressed('output/select_hyperparam/train_accu_' + param.id_simu, a=str(train_acc)) 

    # accuracy validation
    val_acc = hist.history['val_acc']
    np.savez_compressed('output/select_hyperparam/val_accu_' + param.id_simu, a=str(val_acc))

    # loss entrainement
    train_loss = hist.history['loss']
    np.savez_compressed('output/select_hyperparam/train_loss_' + param.id_simu, a=str(train_loss))

    # loss validation
    val_loss = hist.history['val_loss']
    np.savez_compressed('output/select_hyperparam/val_loss_' + param.id_simu, a=str(val_loss))

    # accuracy test
    score = model.evaluate(test_dataset, test_target, verbose=2)
    np.savez_compressed('output/select_hyperparam/test_score_'+ param.id_simu, a=str(score))

    # predictions et representation
    pred = model.predict(test_dataset)

    np.savez_compressed('output/select_hyperparam/pred_' + param.id_simu, a= str(pred))

    # plt.clf()

    param = csv_manager.get_parameters()
