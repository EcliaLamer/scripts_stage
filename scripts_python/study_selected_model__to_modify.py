import numpy as np
import matplotlib.pyplot as plt
import glob
from os.path import basename
import scipy
import scipy.stats
from scipy.stats import kstest
import seaborn as sns
import statsmodels.api as sm
import pylab

from fonctions_stats import descriptives_stats
from fonctions_stats import detect_outliers_iqr

# les chemins 'path_target' et celui utilise pour generer 'pred_list' doivent etre change selon
# le cas etudie. 

path_target = 'output_conv/data_recolte_calculcan_hyperparm_test/test_target.npz'
data_target = np.load(path_target, allow_pickle=True)
data_target = data_target['a']
print(len(data_target))

pred_list = list(glob.glob('output_conv/data_recolte_calculcan_hyperparm_test/resultat_pred/*.npz'))
print(len(pred_list))

def get_nb_pred(path_data):
    name_data = basename(path_data) 
    name_data_without_extension = name_data.split('.')[0]
    name_model = name_data_without_extension.split('_')[1]
    nb_model = int(name_model)

    return nb_model

pred_list = sorted(pred_list, key=get_nb_pred)
center = np.array([0.5,0.5])

num_model = 849
path = pred_list[849]

pred = np.load(path, allow_pickle=True)
pred = pred['a']
err = data_target - pred


plt.title('Histogramme 2D des différences entre les targets et les prédictions')
plt.hist2d(err[:,0], err[:,1], bins=(100, 100), cmap=plt.cm.jet)
plt.colorbar()
plt.show()
plt.clf()

# Simulate data from a bivariate Gaussian
mean = np.mean(err, axis=0)
print('mean :',mean)
cov = np.cov(err, rowvar=0)
print('cov', cov)

n = 50000
rng = np.random.RandomState(0)
print(rng)
x, y = rng.multivariate_normal(mean, cov, n).T

# Draw a combo histogram and scatterplot with density contours
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=err[:,0], y=err[:,1], s=5)
sns.histplot(x=err[:,0], y=err[:,1], bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)
plt.title('Fit de la loi normale pour le modele ' + str(get_nb_pred(path)))
plt.show()
plt.clf()


list_dist_target_vs_pred = []
list_dist_center_pred = []
list_dist_center_targ = []
# list_diff_dist = []
list_dist_center_l1 = []
# list_err_centre_pred = []
# list_err_centre_targ = []

for k in range(len(data_target)):
    # calcul de la distance entre la prediction et la vrai valeur
    dist = np.sqrt(np.square(data_target[k][0]-pred[k][0]) + np.square(data_target[k][1]-pred[k][1]))
    list_dist_target_vs_pred.append(dist)

    # calcul de la distance entre la vrai valeur et le centre (0.5,0.5) de la map
    dist_center_targ = np.sqrt(np.square(center[0]-data_target[k][0]) + np.square(center[1]-data_target[k][1]))
    list_dist_center_targ.append(dist_center_targ)

    # calcul de la distance entre la prediction et le centre (0.5,0.5) de la map
    dist_center_pred = np.sqrt(np.square(center[0]-pred[k][0]) + np.square(center[1]-pred[k][1]))
    list_dist_center_pred.append(dist_center_pred)

    # calcul de la distance norme L1 entre la vraie valeur et le centre (0.5,0.5) de la map
    dist_center_l1 = np.sqrt(np.abs(center[0]-pred[k][0]) + np.abs(center[1]-pred[k][1]))
    list_dist_center_l1.append(dist_center_pred)




array_dist_target_vs_pred = np.array(list_dist_target_vs_pred)
array_dist_center_pred = np.array(list_dist_center_pred)
array_dist_center_targ = np.array(list_dist_center_targ)
array_dist_center_l1 = np.array(list_dist_center_l1)



# Ici on plot pour voir s'il y a un lien entre target eloignée du centre et distance entre target et pred
indice_sort_target_center = np.argsort(array_dist_center_targ)
list_sort_targ_vs_pred = []
for k in indice_sort_target_center:
    list_sort_targ_vs_pred.append(array_dist_target_vs_pred[k])
array_sort_targ_vs_pred = np.array(list_sort_targ_vs_pred)
array_sort_target_vs_center = np.sort(array_dist_center_targ)

# scatter des distances au centre
plt.scatter(np.arange(0.0,len(array_sort_targ_vs_pred)), array_sort_targ_vs_pred, s=0.6, alpha=0.6, label="Distances triees entre targets et predictions")
plt.scatter(np.arange(0.0,len(array_sort_target_vs_center)), array_sort_target_vs_center, s=0.3, alpha=0.6, label="Distance triees entre targets et centre")
# plt.title("Distance target-centre vs distance target-prediction pour le modèle " + str(get_nb_pred(path)))
plt.title("Distance target-centre vs distance target-prédiction pour le modèle ")
plt.show()
plt.clf()


# histo des distances target/pred  ( ne marche pas car on a pris que les val positive a cause du carree)
_ = plt.hist(array_dist_target_vs_pred, bins='auto')
plt.title("Histogramme des distances entre target et prédiction")
plt.show()
plt.clf()


# en faisant la comparaison des distances au centre mais avec tri
indice_sort_target_center = np.argsort(array_dist_center_targ)
list_sort_pred_center= []
for k in indice_sort_target_center:
    list_sort_pred_center.append(array_dist_center_pred[k])
array_sort_pred_center = np.array(list_sort_pred_center)
array_sort_target_vs_center = np.sort(array_dist_center_targ)

# scatter 
plt.scatter(np.arange(0.0,len(array_sort_pred_center)), array_sort_pred_center, s=0.6, alpha=0.6)
plt.scatter(np.arange(0.0,len(array_sort_target_vs_center)), array_sort_target_vs_center,  s=0.6, alpha=0.6)
plt.title("Distances au centre triées ")
plt.show()
plt.clf()


#### Quelques informations sur les outliers

# calcul de la distance max entre la prediction et la vrai valeur
dist_max = np.max(array_dist_target_vs_pred)

# On recupere les 5 distances les plus grande entre la prediction et la vrai valeur et leur indice
indice_sort = np.argsort(array_dist_target_vs_pred)

print('Dans le cas du model', get_nb_pred(path), 'nous avons les résulats suivant :', '\n')
for k in range(1,6):     
    print('la',k,'ieme val de la distance la plus grande est ', array_dist_target_vs_pred[indice_sort[-k]], 'pour la prediction', indice_sort[-k])
    print('les coordonnées de départ sont ', data_target[indice_sort[-k]])
    print('les coordonnées de la prédiction sont', pred[indice_sort[-k]], '\n')

# On recupere les 5 distances les plus grande entre la vraie valeur et le centre et leur indice
indice_sort_center = np.argsort(array_dist_center_targ)


print('Les 15 plus mauvaises prédictions sont :', indice_sort[-15:])
print('les 5 vraies valeurs les plus éloignées du centre sont :', indice_sort_center[-5:])


count_true = 0
for elmt in indice_sort[-15:]:
    if elmt in indice_sort_center[-1000:]:
        count_true +=1
print('le nombre des 15 ouliers en lien avec la distance de la target au centre est ', count_true)


# On récupère les indexes des données test dans l'ensemble des données de départ
indexes_test = np.load("input_conv/test_index.npy")

for k in indice_sort[-15:]:
    print('l indice du', k, ' premier outlier dans le dataset de depart est ' , indexes_test[k])


# la 2694e donnee est un outlier. Ainsi on regarde l image de la maladie dans ce cas
path_to_data_for_outlier_2694 = 'output_conv/data_recolte_calculcan_hyperparm_test/input/data/data_process2.npy'
data_2000_to_2999 = np.load(path_to_data_for_outlier_2694)


plt.imshow(data_2000_to_2999[694],vmin=0, vmax=3)
plt.show()
plt.clf()


# Autres tests pour les outliers

sns.boxplot(array_dist_target_vs_pred)
plt.title("Boxplot des distances")
plt.show()
plt.clf()


def descriptives_stats(donnee):
    print("le nombre de donnees est de :", len(donnee))
    print("la valeur maximale est de :", np.max(donnee))
    print("la valeur minimale est de :", np.min(donnee))
    print("la moyenne des donnees est de :" , np.mean(donnee))
    print("l ecart-type des donnees est de :" , np.std(donnee))
    print("la mediane est :", np.median(donnee))
    first_quant = np.quantile(donnee, 0.25)
    print("le premier quantile est de :" , first_quant)
    third_quant = np.quantile(donnee, 0.75)
    print("le troisieme quantile est de :",  third_quant)
    iqr = third_quant - first_quant
    print("l ecart interquartile est de :", iqr)
    return first_quant, third_quant, iqr

def detect_outliers_iqr(donnee):
    first_quant , third_quant, iqr = descriptives_stats(donnee)
    lower_bound = first_quant - 1.5 * iqr
    upper_bound = third_quant + 1.5 * iqr
    outliers = donnee[( donnee < lower_bound ) | (donnee > upper_bound)]
    ind_outliers = np.where((( donnee < lower_bound ) | (donnee > upper_bound)))
    data_without_outliers = donnee[( donnee >= lower_bound ) & (donnee <= upper_bound)]
    return outliers, data_without_outliers, ind_outliers

outliers, dist_without_outliers, ind_outliers = detect_outliers_iqr(array_dist_target_vs_pred)
np.save( "C:/Users/fviar/Desktop/stage_alice/stage_laptop/Calcul_canada/images_model_849/dist_sans_out.npy", dist_without_outliers)

print("le nombre d outliers par cette methode est de :", len(outliers))
print("\n la proportion outliers est de :", len(outliers)*100 / len(array_dist_target_vs_pred))



for k in ind_outliers[-30:]:
    print('l indice du', k, '  outlier dans le dataset de depart est ' , indexes_test[k])




sns.boxplot(dist_without_outliers)
plt.title("Boxplot des distances sans les outliers")
plt.show()
plt.clf()

descriptives_stats(dist_without_outliers)

#  On recupere nos donnees sans les valeurs des indices des outliers
data_target_without_outliers = np.delete(data_target, ind_outliers, 0)
data_pred_without_outliers = np.delete(pred, ind_outliers, 0)


err_without_outliers = data_target_without_outliers - data_pred_without_outliers
plt.title('Histogramme 2D des différences entre les targets et les predictions sans les outliers ')
plt.hist2d(err_without_outliers[:,0], err_without_outliers[:,1], bins=(100, 100), cmap=plt.cm.jet)
plt.colorbar()
plt.show()
plt.clf()

# fit loi normale
mean_without_outliers = np.mean(err_without_outliers, axis=0)
print('mean :',mean_without_outliers)
cov_without_outliers = np.cov(err_without_outliers, rowvar=0)
print('cov', cov_without_outliers)

n = 50000
rng = np.random.RandomState(0)
print(rng)
x, y = rng.multivariate_normal(mean_without_outliers, cov_without_outliers, n).T

    # Draw a combo histogram and scatterplot with density contours
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=err_without_outliers[:,0], y=err_without_outliers[:,1], s=5)
sns.histplot(x=err_without_outliers[:,0], y=err_without_outliers[:,1], bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)
plt.title('Fit de la loi normale pour le modele sans outliers')
plt.show()
plt.clf()

# On calcul les nouvelles distances au centre

dist_center_pred_without_outliers = np.delete(array_dist_center_pred, ind_outliers, 0)
dist_center_targ_without_outliers = np.delete(array_dist_center_targ, ind_outliers, 0)

print(len(dist_center_pred_without_outliers), len(dist_center_targ_without_outliers))
# On les trie

# en faisant la comparaison des distances au centre mais avec tri
indice_sort_target_center_without_outliers = np.argsort(dist_center_targ_without_outliers)

list_sort_pred_center_without_outliers = []
for k in indice_sort_target_center_without_outliers:
    list_sort_pred_center_without_outliers.append(dist_center_pred_without_outliers[k])
array_sort_pred_center_without_outliers = np.array(list_sort_pred_center_without_outliers)
array_sort_target_vs_center_without_outliers = np.sort(dist_center_targ_without_outliers)

    # scatter 
plt.scatter(np.arange(0.0,len(array_sort_pred_center_without_outliers)), array_sort_pred_center_without_outliers, s=0.6, alpha=0.6)
plt.scatter(np.arange(0.0,len(array_sort_target_vs_center_without_outliers)), array_sort_target_vs_center_without_outliers,  s=0.6, alpha=0.6)
plt.title("Distances au centre triees sans outliers ")
plt.show()
plt.clf()

#  on regarde la logit des coordonnées des erreurs avec et sans outliers
err_norm = (err + 1) / 2

logit_err_x = np.log(err_norm[:,0]/(1-err_norm[:,0]))
logit_err_y = np.log(err_norm[:,1]/(1-err_norm[:,1]))


print("avec kolmo")
print(kstest(logit_err_x, 'norm'))

plt.hist(logit_err_x, density=True)
plt.show()

err_without_outliers_norm = (err_without_outliers + 1) / 2
logit_err_x_without_outliers_norm = np.log(err_without_outliers_norm[:,0]/(1 - err_without_outliers_norm[:,0]))
logit_err_y_without_outliers_norm = np.log(err_without_outliers_norm[:,1]/(1 - err_without_outliers_norm[:,1]))

plt.hist(logit_err_x_without_outliers_norm[:1000], bins=30, density=True)
plt.show()
plt.hist(logit_err_y_without_outliers_norm[:1000], bins=30, density=True)
plt.show()

print(scipy.stats.normaltest(logit_err_x_without_outliers_norm))
print(scipy.stats.normaltest(logit_err_y_without_outliers_norm))

scipy.stats.probplot(logit_err_x_without_outliers_norm,dist="norm", plot=pylab)
pylab.show()


# on fit la gamma sur les distances target-prédiction avec et sans outliers
a, loc, scale = scipy.stats.gamma.fit(array_dist_target_vs_pred)
X = np.arange(-0.1,.5,0.01)


a_bis, loc_bis, scale_bis = scipy.stats.gamma.fit(dist_without_outliers)

plt.hist(array_dist_target_vs_pred, bins=30, density=True)
plt.plot(X,scipy.stats.gamma.pdf(X,a,loc=loc,scale=scale))
plt.show()
plt.clf()
