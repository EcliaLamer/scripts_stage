import numpy as np

ARR_WEEKLY_MORTALITY = []
for x in [0.6, .4, .3, .3, .3, .6, .6]:
    for _ in range(52):
        ARR_WEEKLY_MORTALITY.append(x)
ARR_WEEKLY_MORTALITY = np.array(ARR_WEEKLY_MORTALITY)
ARR_WEEKLY_MORTALITY = 1 - (1 - ARR_WEEKLY_MORTALITY) ** (1. / 52.)

ARR_NB_WEEK_INF = np.array([1, 2, 3, 4])
ARR_PROB_WEEK_INF = np.array([0.25, 0.25, 0.25, 0.25])