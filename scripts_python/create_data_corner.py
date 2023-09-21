import multiprocessing as mlp
from sampy.agent.builtin_agent import BasicMammal
from sampy.graph.builtin_graph import SquareGridWithDiag
from sampy.disease.single_species.builtin_disease import ContactCustomProbTransitionPermanentImmunity
from constant_paper import ARR_WEEKLY_MORTALITY, ARR_NB_WEEK_INF, ARR_PROB_WEEK_INF
import numpy as np
import matplotlib.pyplot as plt
import random

print('ici on cree une banque de donnees dans des carrees aux 4 coins de la map')

number_cores = 40
rng_seed = 7521

def worker(id_process, nb_process, rng_seed):
    np.random.seed(rng_seed + id_process)
    list_one_of_simu = []
    list_mid_start = []
    for simu in range(250):

        # path to the population csv
        path_pop_csv = 'output/population.csv'

        # path to a folder where to save the data
        path_output_folder = 'output/output_corner/data'
        # path to a folder where to save the start_vertice
        path_output_folder_start = 'output/output_corner/start'

        # create the landscape
        my_graph = SquareGridWithDiag(shape=(100, 100))
        my_graph.create_vertex_attribute('K', 15.)
        # create the population object
        agents = BasicMammal(graph=my_graph)
        agents.load_population_from_csv(path_pop_csv)
        # create a disease and initiate infection at the center of the map

        tuple_for_vertex = np.random.randint(0, 3, size=2)
        if tuple_for_vertex[1] == 2:
            neig_tuple_for_vertex = [tuple_for_vertex[0], tuple_for_vertex[1] - 1]
        else :
            neig_tuple_for_vertex = [tuple_for_vertex[0], tuple_for_vertex[1] + 1] 

        list_of_extrem_corner = [[0, 0], [0, 97], [97, 0], [97, 97]]
        one_of_extrem_corner = random.choice(list_of_extrem_corner)
        start_vertice = one_of_extrem_corner + tuple_for_vertex
        start_vertice = tuple(start_vertice)
        neig_vertice = one_of_extrem_corner + neig_tuple_for_vertex
        neig_vertice = tuple(neig_vertice)
        list_start_vertices = [start_vertice, neig_vertice]

        disease = ContactCustomProbTransitionPermanentImmunity(host=agents, disease_name='disease') 
        arr_new_contamination = disease.contaminate_vertices(list_start_vertices, .5) #voir ou il y a voisin//oir comment virer simu sans rage
        disease.initialize_counters_of_newly_infected(arr_new_contamination, ARR_NB_WEEK_INF, ARR_PROB_WEEK_INF)
        
        nb_year_simu = 3
        
        list_inf_pic = []
        for year in range(nb_year_simu):
            for week in range(52):
                list_inf_pic.append(my_graph.convert_1d_array_to_2d_array(disease.count_nb_status_per_vertex('inf',
                                                                        position_attribute='territory')))
                agents.tick()
                my_graph.tick()
                disease.tick()

                agents.kill_too_old(52 * 6 - 1)
                agents.natural_death_orm_methodology(ARR_WEEKLY_MORTALITY, ARR_WEEKLY_MORTALITY)
                agents.kill_children_whose_mother_is_dead(11)

                agents.mov_around_territory(0.5, condition=agents.df_population['age'] >= 11)
                arr_new_infected = disease.contact_contagion(0.1, return_arr_new_infected=True)
                disease.initialize_counters_of_newly_infected(arr_new_infected, ARR_NB_WEEK_INF, ARR_PROB_WEEK_INF)
                disease.transition_between_states('con', 'death', proba_death=0.8)
                disease.transition_between_states('con', 'imm')
                disease.transition_between_states('inf', 'con',
                                                arr_nb_timestep=np.array([1, 2]),
                                                arr_prob_nb_timestep=np.array([.5, .5]))

                if week == 15:
                    agents.find_random_mate_on_position(1., position_attribute='territory')
                if week == 22:
                    agents.create_offsprings_custom_prob(np.array([4, 5, 6, 7, 8, 9]), np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))
                if week == 40:
                    can_move = agents.df_population['age'] > 11
                    agents.dispersion_with_varying_nb_of_steps(np.array([1, 2, 3, 4]), np.array([.25, .25, .25, .25]),
                                                    condition=can_move)

        list_one_of_simu.append(random.choice(list_inf_pic))
        list_mid_start.append([(start_vertice[0] + neig_vertice[0]) / 2. , (start_vertice[1] + neig_vertice[1]) / 2.])

    array_frame_one_of_simu = np.asfarray(list_one_of_simu)
    arr_mid_start_vertices = np.asfarray(list_mid_start)

    np.save(path_output_folder +'_process' +str(id_process) +'.npy', array_frame_one_of_simu)
    np.save(path_output_folder_start +'_process' +str(id_process) +'.npy', arr_mid_start_vertices)#transfo float+norm

    print("coucou, je suis process", id_process, 'sur', nb_process)


if __name__ == '__main__':
    list_jobs = []
    for id_proc in range(number_cores):
        list_jobs.append(mlp.Process(target=worker, args=(id_proc, number_cores, rng_seed)))
        list_jobs[-1].start()

    for p in list_jobs:
        p.join()