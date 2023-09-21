# import nlmpy.nlmpy as nlmpy
from nlmpy_temp import mpd, classifyArray
import numpy as np
import multiprocessing as mlp
from sampy.agent.builtin_agent import BasicMammal
from sampy.graph.builtin_graph import SquareGridWithDiag
from sampy.disease.single_species.builtin_disease import ContactCustomProbTransitionPermanentImmunity
from constant_paper import ARR_WEEKLY_MORTALITY, ARR_NB_WEEK_INF, ARR_PROB_WEEK_INF
import matplotlib.pyplot as plt
import random
import time

print('On cree des donnees avec des maps de ressource obtenues avec NLMPY')
print('Seed utilise : 1')

number_cores = 40
rng_seed = 1

def worker(id_process, nb_process, rng_seed):
    np.random.seed(rng_seed + id_process)

    nb_simu = 100_000
    size_batch = 100
    count_batch_created = 0

    list_one_of_simu = []
    list_starts = []
    list_map_resources = []
    list_date_of_extract = []

    for simu in range(nb_simu): 

        # path to the population csv
        path_pop_csv = 'output/population.csv'

        # path to a folder where to save the data
        path_output_folder = 'output/data_ressources_autocorr_1/data'
        # path to a folder where to save the start_vertice
        path_output_folder_start = 'output/data_ressources_autocorr_1/start'
        path_output_folder_resources = 'output/data_ressources_autocorr_1/resources'
        path_output_folder_date = 'output/data_ressources_autocorr_1/dates'

        # create the landscape
        my_graph = SquareGridWithDiag(shape=(100, 100))

        # --------------------------------------------------------------------------------------------------------------
        # on veut 4 classes de cellules sur notre paysage
        # on va arbitrairement choisir une valeur de k par classe

        class_of_infect_start = 3.
        dict_class_to_K = {0.: 0., 1.: 5., 2.: 10., 3.: 20.}
        auto_correlation = 1  # possiblement a faire varier

        map_with_deform = mpd(100, 100, auto_correlation)
        map_with_class = classifyArray(map_with_deform, [1, 0.5, 0.5, 1])

        # on convertit ca en carte
        map_with_k = np.zeros(map_with_class.shape, dtype=float)

        for class_as_float, k in dict_class_to_K.items():
            map_with_k += k * (map_with_class == class_as_float)

        my_graph.create_attribute_from_2d_array('K', map_with_k)
     
        # --------------------------------------------------------------------------------------------------------------

        # create the population object
        agents = BasicMammal(graph=my_graph) 
        agents.load_population_from_csv(path_pop_csv)

        # --------------------------------------------------------------------------------------------------------------
        # on change radicalement ce qu'on fait pour la maladie, ici on va juste creer la maladie sans infecter personne
        disease = ContactCustomProbTransitionPermanentImmunity(host=agents, disease_name='disease')
        # --------------------------------------------------------------------------------------------------------------

        nb_year_simu = 8
        list_inf_pic = []
        for year in range(nb_year_simu):
            for week in range(52):

                # ------------------------------------------------------------------------------------------------------
                # on infecte au debut de l'annee 6 (5 en comptant en python)
                if year == 5 and week == 0:
                    # on doit choisir une cellule ou commencer l'infection, OR on a beaucoup de cellules qui ont un
                    # K tres faible. Donc on ne veut surtout pas prendre une cellule comme ca comme debut d'infection
                    # il faut prendre une cellule avec un k de 20 au hasard.
                    list_vertice_infection = []
                    for i in range(100):
                        for j in range(100):
                            if map_with_class[i][j] == class_of_infect_start:
                                list_vertice_infection.append((i,j))

                    list_vertex_infection = [random.choice(list_vertice_infection)]  # comme cette cellule doit avoir un k de 20, en prendre un seul
                                                         # devrait suffir
                    disease.simplified_contaminate_vertices(list_vertex_infection, 1.,
                                                            ARR_NB_WEEK_INF, ARR_PROB_WEEK_INF)

                # ------------------------------------------------------------------------------------------------------

                if year >= 5: # on extrait des images de l'infection uniquement quand on a lance la maladie

                    list_inf_pic.append(my_graph.convert_1d_array_to_2d_array(disease.count_nb_status_per_vertex('inf',
                                                                              position_attribute='territory')))

                agents.tick()
                my_graph.tick()
                disease.tick()

                agents.kill_too_old(52 * 6 - 1)
                agents.natural_death_orm_methodology(ARR_WEEKLY_MORTALITY, ARR_WEEKLY_MORTALITY)
                agents.kill_children_whose_mother_is_dead(11)

                agents.mov_around_territory(0.5, condition=agents.df_population['age'] >= 11)
                # ------------------------------------------------------------------------------------------------------
                arr_new_infected = disease.contact_contagion(0.1, return_arr_new_infected=True)
                disease.initialize_counters_of_newly_infected(arr_new_infected, ARR_NB_WEEK_INF, ARR_PROB_WEEK_INF)
                disease.transition_between_states('con', 'death', proba_death=0.8)
                disease.transition_between_states('con', 'imm')
                disease.transition_between_states('inf', 'con',
                                                    arr_nb_timestep=np.array([1, 2]),
                                                    arr_prob_nb_timestep=np.array([.5, .5]))
                # ------------------------------------------------------------------------------------------------------

                if week == 15:
                    agents.find_random_mate_on_position(1., position_attribute='territory')
                if week == 22:
                    agents.create_offsprings_custom_prob(np.array([4, 5, 6, 7, 8, 9]), np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))
                if week == 40:
                    can_move = agents.df_population['age'] > 11
                    agents.dispersion_with_varying_nb_of_steps(np.array([1, 2, 3, 4]), np.array([.25, .25, .25, .25]),
                                                    condition=can_move)

        week_of_extraction = random.randint(1,len(list_inf_pic)-1) # On ne prend pas la premiere semaine de propagation de la maladie

        list_one_of_simu.append(list_inf_pic[week_of_extraction])        
        # -------------------------------------------------------------------------------------------------------------
        list_starts.append(list_vertex_infection)
        # -------------------------------------------------------------------------------------------------------------
        list_map_resources.append(map_with_k)
        # -------------------------------------------------------------------------------------------------------------  
        list_date_of_extract.append(week_of_extraction)      

        # on ecrit les data si on a atteint la taille de batch
        if len(list_one_of_simu) >= size_batch:
            array_frame_one_of_simu = np.asfarray(list_one_of_simu)
            arr_starts_vertices = np.asfarray(list_starts)
            arr_map_resources = np.asfarray(list_map_resources)
            arr_date_of_extract = np.asfarray(list_date_of_extract)

            np.save(path_output_folder + '_batch' + str(count_batch_created) + '_process' +str(id_process) +'.npy', array_frame_one_of_simu)
            np.save(path_output_folder_start + '_batch' + str(count_batch_created) +'_process' +str(id_process) +'.npy', arr_starts_vertices)
            np.save(path_output_folder_resources + '_batch' + str(count_batch_created) +'_process' +str(id_process) +'.npy', arr_map_resources)
            np.save(path_output_folder_date + '_batch' + str(count_batch_created) +'_process' +str(id_process) +'.npy', arr_date_of_extract)

            count_batch_created += 1
            list_one_of_simu = []
            list_map_resources = []
            list_starts = []

    if not list_starts:
        array_frame_one_of_simu = np.asfarray(list_one_of_simu)
        arr_starts_vertices = np.asfarray(list_starts)
        arr_map_resources = np.asfarray(list_map_resources)
        arr_date_of_extract = np.asfarray(list_date_of_extract)

        np.save(path_output_folder + '_batch' + str(count_batch_created) + '_process' +str(id_process) +'.npy', array_frame_one_of_simu)
        np.save(path_output_folder_start + '_batch' + str(count_batch_created) +'_process' +str(id_process) +'.npy', arr_starts_vertices)
        np.save(path_output_folder_resources + '_batch' + str(count_batch_created) +'_process' +str(id_process) +'.npy', arr_map_resources)
        np.save(path_output_folder_date + '_batch' + str(count_batch_created) +'_process' +str(id_process) +'.npy', arr_date_of_extract)

    print("coucou, je suis process", id_process, 'sur', nb_process)


if __name__ == '__main__':
    list_jobs = []
    for id_proc in range(number_cores):
        list_jobs.append(mlp.Process(target=worker, args=(id_proc, number_cores, rng_seed)))
        list_jobs[-1].start()

    for p in list_jobs:
        p.join()