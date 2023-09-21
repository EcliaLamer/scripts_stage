import tensorflow as tf
import numpy as np
import glob
from os.path import basename, splitext
import time
from tensorflow.keras import Input as KInput
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Flatten, Dropout, concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ATTENTION, ce script ne peut pas etre utilise ici en l'etat,
# car il repose sur une quantite enorme de donnee genere grace au
# supercalculateur Beluga (Calcul Canada)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print('Nous testons ici notre reseau de neurones en considerant des cartes de ressources et en utilisant un tf.data.dataset pour gerer les donnees')

def generator_read_files(list_path_data, list_path_ressources, list_path_target):
    """
    We assume the lists are of the same length and already sorted.
    """
    for i in range(len(list_path_data)):
        data = 0.02 * np.asfarray(np.load(list_path_data[i]))
        ressource = 0.02 * np.asfarray(np.load(list_path_ressources[i]))
        target = 0.01 * np.asfarray(np.load(list_path_target[i]))
        target = target[:, 0, :]
        for j in range(data.shape[0]):
            yield (data[j, :, :], ressource[j, :, :]),  target[j, :]



start_time = time.time()

# Fonctions pour le traitement des datas brutes
# Fonction pour ordonner les donnees et les target
def get_nb_process_and_batch_from_path(path_data):
    name_data = basename(path_data) # recup le nom de l'image
    name_data_without_extension = name_data.split('.')[0]
    name_process_with_process = name_data_without_extension.split('_')[2]
    name_process = name_process_with_process.replace("process","")
    nb_process = int(name_process)
    name_batch_with_batch = name_data_without_extension.split('_')[1]
    name_batch= name_batch_with_batch.replace("batch","")
    nb_batch = int(name_batch)
    return nb_batch, nb_process

# Fonction pour recupere pour chaque process le nombre de batch partages entre tous
def max_nb_batch_for_all_process(liste, nb_process_max):
    nb_batch_list = []
    new_liste = []
    for i in range(nb_process_max):
        process_list = []
        for name in liste:
            name_base = basename(name)
            num_of_process = name_base.split('.')[0].split('_')[2].replace("process","")
            process_nb = int(num_of_process)
            if process_nb == i:
                process_list.append(name_base)
        number_of_batch = len(process_list)
        nb_batch_list.append(number_of_batch)
    max_batch = min(nb_batch_list) - 1
    print("max des batchs", max_batch)
    for name in liste:
        name_base = basename(name)
        num_of_batch = name_base.split('.')[0].split('_')[1].replace("batch","")
        batch_nb = int(num_of_batch)
        if batch_nb <= max_batch:
            new_liste.append(name)
    return new_liste

# On importe les datas brutes et on applique nos fonction

list_folder = ['data_ressources_seed_1789', 'data_ressources_seed_42', 'data_ressources_seed_421']

data_list = []
target_list = []
resources_list = []
dates_list = []
nb_process = 40

for folder in list_folder:
    
# on recup lelien des donnees de calcul canada
    name_data_list = list(glob.glob(folder +'/data*.npy'))
    name_target_list = list(glob.glob(folder +'/start*.npy'))
    name_resources_list = list(glob.glob(folder +'/resources*.npy'))
    name_dates_list = list(glob.glob(folder +'/dates*.npy'))


# on applique la fonction de tri

    name_data_list = sorted(name_data_list, key=get_nb_process_and_batch_from_path)
    name_target_list = sorted(name_target_list, key=get_nb_process_and_batch_from_path)
    name_resources_list = sorted(name_resources_list, key=get_nb_process_and_batch_from_path)
    name_dates_list = sorted(name_dates_list, key=get_nb_process_and_batch_from_path)


# on applique la fonction du max de batch
    data_list = data_list + max_nb_batch_for_all_process(name_data_list, nb_process)
    target_list = target_list + max_nb_batch_for_all_process(name_target_list, nb_process)
    resources_list = resources_list + max_nb_batch_for_all_process(name_resources_list, nb_process)
    dates_list = dates_list + max_nb_batch_for_all_process(name_dates_list, nb_process)

dataset = tf.data.Dataset.from_generator(generator_read_files,
                                         args=[data_list, resources_list, target_list],
                                         output_signature=((tf.TensorSpec(shape=(100, 100), dtype=tf.float32),
                                                            tf.TensorSpec(shape=(100, 100), dtype=tf.float32)),
                                                           tf.TensorSpec(shape=(2), dtype=tf.float32)))


# creation du modele
# # Reseau de neurones

# # Initialisation du modèle

# # # Construct the input layer with no definite frame size.
input_infected_map = KInput(shape=(100, 100, 1), name='input_inf')
# maintenant la meme chose mais pour la carte des ressources
input_ressource_map = KInput(shape=(100, 100, 1), name="input_ressource")


# ensuite on aligne les layers de convolution, maxpooling, et on fini par le flatten (je te laisse gerer l'architecture
# de ca en prenant le model qui a bien marche)
inf_map_1 = Conv2D(32, (2, 2), padding='same', activation='relu')(input_infected_map)
inf_map_2 = MaxPooling2D(pool_size=(2, 2))(inf_map_1)
inf_map_3 = Conv2D(32, (2, 2), activation='relu')(inf_map_2) # pas besoin de changer le nom entre les etapes, mais je prefere
inf_map_4 = MaxPooling2D(pool_size=(2, 2))(inf_map_3)
inf_map_drop_1 = Dropout(0.2)(inf_map_4)

inf_map_5 = Conv2D(64, (3, 3), padding='same', activation = 'relu')(inf_map_drop_1)
inf_map_6 = MaxPooling2D(pool_size=(2, 2))(inf_map_5)
inf_map_7 = Conv2D(64, (3, 3), activation='relu')(inf_map_6) # pas besoin de changer le nom entre les etapes, mais je prefere
inf_map_8 = MaxPooling2D(pool_size=(2, 2))(inf_map_7)
inf_map_drop_2 = Dropout(0.3)(inf_map_8)

inf_map_flat = Flatten()(inf_map_drop_2)



ressource_map_1 = Conv2D(32, (2, 2), padding='same', activation = 'relu')(input_ressource_map)
ressource_map_2 = MaxPooling2D(pool_size=(2, 2))(ressource_map_1)
ressource_map_3 = Conv2D(32, (2, 2), activation='relu')(ressource_map_2)
ressource_map_4 = MaxPooling2D(pool_size=(2, 2))(ressource_map_3)
ressource_map_drop_1 = Dropout(0.2)(ressource_map_4)

ressource_map_5 = Conv2D(64, (3, 3), padding='same', activation = 'relu')(inf_map_drop_1)
ressource_map_6 = MaxPooling2D(pool_size=(2, 2))(ressource_map_5)
ressource_map_7 = Conv2D(64, (3, 3), activation='relu')(ressource_map_6)
ressource_map_8 = MaxPooling2D(pool_size=(2, 2))(ressource_map_7)
ressource_map_drop_2 = Dropout(0.3)(ressource_map_8)

ressource_map_flat = Flatten()(ressource_map_drop_2)


# maintenant on concatene le tout
x = concatenate([inf_map_flat, ressource_map_flat])

# on ajoute les couches denses finales
x = Dense(units = 64, activation = 'relu')(x)
x = Dense(units = 64, activation = 'relu')(x)
output_start = Dense(units = 2, activation='sigmoid', name='output_start')(x)


# maintenant on definit le model en lui precisant les entree et "les" sortie"s"
# model = Model(inputs=[input_infected_map, input_ressource_map], outputs=[output])
model = Model(inputs=[input_infected_map, input_ressource_map], outputs=[output_start])


model.compile(
    loss='mean_squared_error', optimizer='adam',metrics=['accuracy']
)

model.summary()
# plot_model(model, "multi_input_and_output_model.png", show_shapes=True)


epochs = 10
batch_size = 64


hist = model.fit(dataset.batch(batch_size).prefetch(3),
    epochs=epochs,
    verbose=2)


# Enregistrement des scores

output_path = 'output_for_nn_with_all_resources/'

    # accuracy entrainement
train_acc = hist.history['accuracy']
np.savez_compressed(output_path +'acc/train_accu' , a=train_acc)

    # loss entrainement
train_loss = hist.history['loss']
np.savez_compressed(output_path + 'loss/train_loss' , a=train_loss)




# On recupere et traite nos donnees d evaluation 

name_data_list_test = list(glob.glob('data_ressources_seed_1/data*.npy'))
name_target_list_test = list(glob.glob('data_ressources_seed_1/start*.npy'))
name_resources_list_test = list(glob.glob('data_ressources_seed_1/resources*.npy'))
name_dates_list_test = list(glob.glob('data_ressources_seed_1/dates*.npy'))

name_data_list_test = sorted(name_data_list_test, key=get_nb_process_and_batch_from_path)
name_target_list_test = sorted(name_target_list_test, key=get_nb_process_and_batch_from_path)
name_resources_list_test = sorted(name_resources_list_test, key=get_nb_process_and_batch_from_path)
name_dates_list_test = sorted(name_dates_list_test, key=get_nb_process_and_batch_from_path)

data_list_test = max_nb_batch_for_all_process(name_data_list_test, nb_process)
target_list_test = max_nb_batch_for_all_process(name_target_list_test, nb_process)
resources_list_test = max_nb_batch_for_all_process(name_resources_list_test, nb_process)
dates_list_test = max_nb_batch_for_all_process(name_dates_list_test, nb_process)

list_datas = []
for filename in data_list_test:
    data = 0.02 * np.asfarray(np.load(filename))
    list_datas.append(data)
test_dataset = np.concatenate((list_datas),axis=0)

## pour les ressources

list_resources = []
for filename in resources_list_test:
    resources = 0.02 * np.asfarray(np.load(filename))
    list_resources.append(resources)
test_resources = np.concatenate((list_resources),axis=0)


## pour les targets

list_starts = []
for filename in target_list_test:
    start_float = 0.01 * np.asfarray(np.load(filename))
    list_starts.append(start_float)
array_targets = np.concatenate((list_starts),axis=0)

test_target = array_targets[:, 0, :]
np.savez_compressed(output_path + 'test_target', a=test_target)

print("Testing Dataset Shapes: " + str(test_dataset.shape) + ", " + str(test_target.shape) + "," + str(test_resources.shape))


#     # loss and accuracy test
score = model.evaluate({'input_inf': test_dataset, 'input_ressource': test_resources}, {'output_start': test_target}, verbose=2)
    #test_acc = score[1]
    #test_loss = score[0]
np.savez_compressed(output_path + 'score', a=score)
print('Le score du model est ', score)


    # predictions

pred = model.predict({'input_inf': test_dataset, 'input_ressource': test_resources})

np.savez_compressed(output_path + 'pred', a= pred)

print("Le processus a pris", time.time()-start_time)