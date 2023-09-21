import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

import glob
import os
from os.path import basename
import time

# ----------------------------------------------------------------
# parametres
nb_repetition_train_nn = 50
# ----------------------------------------------------------------

# create folder if needed
path_folder_output = 'output_nn_k17_5_repetition'
if os.path.exists(path_folder_output):
    raise ValueError("Already a Folder at " + path_folder_output)
os.mkdir(path_folder_output)

print("Ici on entraine", nb_repetition_train_nn, "nn avec un k de 17.5")


start_time = time.time()

# On importe les datas brutes
# on recup lelien des donnees de calcul canada
name_data_list = list(glob.glob('data_and_output_k17_5/data/data*.npy'))
name_target_list = list(glob.glob('data_and_output_k17_5/data/start*.npy'))

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

for k in range(nb_repetition_train_nn):

    # On split en deux ensemble : train et test (par indexing)
    indexes = np.arange(np.shape(array_data_set)[0])
    np.random.seed(460 + k)
    np.random.shuffle(indexes)

    train_index = indexes[: int(0.8 * np.shape(array_data_set)[0])]
    test_index = indexes[int(0.8 * np.shape(array_data_set)[0]) :]
    np.save(path_folder_output + '/test_index_' + str(k) + '.npy', test_index)

    train_dataset = array_data_set[train_index]
    test_dataset = array_data_set[test_index]

    train_target = array_targets[train_index]
    test_target = array_targets[test_index]
    np.savez_compressed(path_folder_output + '/test_target_' + str(k), a= test_target)

    # reseau de neurones

    # Initialisation du modèle
    model = Sequential()

    # Première couche convolutive
    model.add(Conv2D(32, (2, 2), padding='same', input_shape = (100, 100, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Ajout d'une deuxième couche convolutive
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Flatten
    model.add(Flatten())

    # Connexion complète
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dense(units = 2, activation = 'sigmoid'))

    # Compile
    model.compile(
        loss='mean_squared_error', optimizer='adam', metrics=['accuracy']
        )

    #model.summary()

    # Define modifiable training hyperparameters.
    epochs = 10
    batch_size = 64

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
    train_acc = hist.history['accuracy']
    np.savez_compressed(path_folder_output + '/train_accu_' + str(k) , a=train_acc)

        # accuracy validation
    val_acc = hist.history['val_accuracy']
    np.savez_compressed(path_folder_output + '/val_accu_' + str(k) , a=val_acc)

        # loss entrainement
    train_loss = hist.history['loss']
    np.savez_compressed(path_folder_output + '/train_loss_' + str(k) , a=train_loss)

        # loss validation
    val_loss = hist.history['val_loss']
    np.savez_compressed(path_folder_output + '/val_loss_' + str(k), a=val_loss)

        # loss and accuracy test
    score = model.evaluate(test_dataset, test_target, verbose=2)
        #test_acc = score[1]
        #test_loss = score[0]
    np.savez_compressed(path_folder_output + '/score_' + str(k), a=score)


        # predictions
    pred = model.predict(test_dataset)

    np.savez_compressed(path_folder_output + '/pred_' + str(k), a= pred)

print("Le process nn a pris", time.time() - start_time, 'secondes.')