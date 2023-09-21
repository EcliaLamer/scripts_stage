import numpy as np
import matplotlib.pyplot as plt
import glob
from os.path import basename
from operator import itemgetter


score_list = list(glob.glob('output/select_hyperparam/test_score_*.npz'))

# On recupere le numero du model pour trier les paths de nos données
def get_nb_model(path_data):
    name_data = basename(path_data) # recup le nom du fichier
    name_data_without_extension = name_data.split('.')[0]
    name_model = name_data_without_extension.replace('test_score_', '')
    nb_model = int(name_model)

    return nb_model

# on trie
score_list = sorted(score_list, key=get_nb_model)
print("Nombre de scores trouves :", len(score_list))

# on recupere les valeurs extremes des acc et leur modeles associes
model_acc_loss_list =[]
acc_liste = []
loss_liste = []
for num_model, path in enumerate(score_list):
    score = np.load(path, allow_pickle=True)
    score = score['a']
    acc = score[1]
    loss = score[0]
    model_acc_loss_list.append([num_model, acc, loss])
    acc_liste.append(acc)
    loss_liste.append(loss)

print("On verifie la longueur des listes extraites :", len(acc_liste), len(loss_liste), len(model_acc_loss_list))
	
def maximum_indices(liste):
    maxi = liste[0]
    longueur = len(liste)
    model = 0
    for i in range(longueur):
        if liste[i] >= maxi:
            maxi = liste[i]
            model = i

    return maxi, model

def minimum_indices(liste):
    _ , num_model_max = maximum_indices(liste)
    mini = liste[num_model_max]
    longueur=len(liste)
    model = 0
    for i in range(longueur):
        if liste[i] <= mini:
            mini = liste[i]
            model = i

    return mini, model

############### regarder pour la loss


# les loss la plus petite et la plus grandes et leur modeles
max_loss, num_model_max = maximum_indices(loss_liste)
min_loss, num_model_min = minimum_indices(loss_liste)
print('la loss maximale est de', max_loss, 'pour le model', num_model_max,'\n')
print('la loss minimale est de' , min_loss, 'pour le model', num_model_min, '\n' )

# les hyper-parametres de ces deux modeles

list_hyper_param = []
with open('csv_hyper_param_nn_conv.csv', 'r', encoding='UTF8') as csv_param:
    for ligne in csv_param.readlines():
        list_hyper_param.append(ligne)

print('Les hyper parametres pour le model', num_model_max, 'sont',list_hyper_param[num_model_max + 1 ])
print('Les hyper parametres pour le model', num_model_min, 'sont',list_hyper_param[num_model_min + 1 ])

############### regarder pour l'acc


# les accuracy la plus petite et la plus grandes et leur modeles
max_acc, num_model_max = maximum_indices(acc_liste)
min_acc, num_model_min = minimum_indices(acc_liste)
print('l accuracy maximale est de', max_acc, 'pour le model', num_model_max,'\n')
print('l accuracy minimale est de' , min_acc, 'pour le model', num_model_min, '\n' )

# les hyper-parametres de ces deux modeles

list_hyper_param = []
with open('csv_hyper_param_nn_conv.csv', 'r', encoding='UTF8') as csv_param:
    for ligne in csv_param.readlines():
        list_hyper_param.append(ligne)

print('Les hyper parametres pour le model', num_model_max, 'sont',list_hyper_param[num_model_max + 1 ])
print('Les hyper parametres pour le model', num_model_min, 'sont',list_hyper_param[num_model_min + 1 ])



# les dix meilleurs modeles
print(' le nombre de modeles étudiés est de', len(model_acc_loss_list))

# etude de la loss et de l'acc
list_modeles_tries = sorted(model_acc_loss_list, key=lambda u: -itemgetter(2)(u))

print('les dix meilleurs modeles et leur scores sont :')
print(list_modeles_tries[-10:])
print('')
print('les 60 plus mauvais modeles et leur acc sont :')
print(list_modeles_tries[:60])
