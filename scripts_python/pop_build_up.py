from sampy.agent.builtin_agent import BasicMammal
from sampy.graph.builtin_graph import SquareGridWithDiag
from constant_paper import ARR_WEEKLY_MORTALITY

import numpy as np

# path to a folder where to save the population csv
path_output_folder = 'output'

# create the landscape
my_graph = SquareGridWithDiag(shape=(100, 100))
my_graph.create_vertex_attribute('K', 10.)

# create the population object
agents = BasicMammal(graph=my_graph)

# create some agents
dict_new_agents = dict()
dict_new_agents['age'] = [52, 52, 52, 52, 52, 52, 52, 52, 52, 52]
dict_new_agents['gender'] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
dict_new_agents['territory'] = [0, 0, 125, 125, 1021, 1021, 2034, 2034, 6321, 6321]
dict_new_agents['position'] = [0, 0, 125, 125, 1021, 1021, 2034, 2034, 6321, 6321]
agents.add_agents(dict_new_agents)


nb_year_simu = 100
for i in range(nb_year_simu * 52 + 1):

    agents.tick()
    my_graph.tick()

    agents.kill_too_old(52 * 6 - 1)
    agents.natural_death_orm_methodology(ARR_WEEKLY_MORTALITY, ARR_WEEKLY_MORTALITY)
    agents.kill_children_whose_mother_is_dead(11)

    agents.mov_around_territory(0.8)

    if i % 52 == 15:
        agents.find_random_mate_on_position(1., position_attribute='territory')
    if i % 52 == 22:
        agents.create_offsprings_custom_prob(np.array([4, 5, 6, 7, 8, 9]), np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))
    if i % 52 == 40:
        can_move = agents.df_population['age'] > 11
        agents.dispersion_with_varying_nb_of_steps(np.array([1, 2, 3, 4]),
                                                   np.array([.25, .25, .25, .25]),
                                                   condition=can_move)

agents.save_population_to_csv(path_output_folder + '/population.csv')