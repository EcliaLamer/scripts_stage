import numpy as np

def descriptives_stats(donnee):
    """
    Produit les statistiques descriptives a partir d'une array numpy 1D
    """
    print("le nombre de donnees est de :", len(donnee))
    print("la valeur maximale est de :", np.max(donnee))
    print("la valeur minimale est de :", np.min(donnee))
    print("la moyenne des donnees est de :" , np.mean(donnee))
    print("la mediane des donnees est de :" , np.median(donnee))
    print("l ecart-type des donnees est de :" , np.std(donnee))
    first_quant = np.quantile(donnee, 0.25)
    print("le premier quantile est de :" , first_quant)
    third_quant = np.quantile(donnee, 0.75)
    print("le troisieme quantile est de :",  third_quant)
    iqr = third_quant - first_quant
    print("l ecart interquartile est de :", iqr)
    return first_quant, third_quant, iqr

def detect_outliers_iqr(donnee):
    """
    Extrait les outliers a partir d'une array numpy 1D
    """
    first_quant , third_quant, iqr = descriptives_stats(donnee)

    lower_bound = first_quant - 1.5 * iqr
    upper_bound = third_quant + 1.5 * iqr

    outliers = donnee[( donnee < lower_bound ) | (donnee > upper_bound)]

    ind_outliers = np.where((( donnee < lower_bound ) | (donnee > upper_bound)))

    data_without_outliers = donnee[( donnee >= lower_bound ) & (donnee <= upper_bound)]

    return outliers, data_without_outliers, ind_outliers