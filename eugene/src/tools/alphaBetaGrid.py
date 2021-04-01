import numpy as np


def lv_map_points(alpha_min=0, alpha_max=1.2, alpha_steps=100.0, beta_min=-0.2,
                  beta_max=1.2, beta_steps=100.0):
    """
    :param alpha_min: default 0; minimum alpha coordinate
    :param alpha_max: default 1.2; maximum alpha coordinate
    :param alpha_steps: default 100; number of alpha coordinates to visit
    :param beta_min: default -0.2; minimum beta coordinate
    :param beta_max: default 1.2; maximum beta coordinate
    :param beta_steps: default 100; number of beta coordinates to visit
    :return: a list of 4-tuples containing, in order, the growth rate vector,
         the interaction matrix, the alpha coordinate, and the beta coordinate.
    """

    A1 = np.array([(1.0, 2.419, 2.248, 0.0023),
                   (0.001, 1.0, 0.001, 1.3142),
                   (2.3818, 0.001, 1.0, 0.4744),
                   (1.21, 0.5244, 0.001, 1.0)])
    A2 = np.array([(1.0, 0.3064, 2.9141, 0.8668),
                   (0.125, 1.0, 0.3346, 1.854),
                   (1.9833, 3.5183, 1.0, 0.001),
                   (0.6986, 0.9653, 2.1232, 1.0)])
    A3 = np.array([(1.0, 3.6981, 1.4368, 0.0365),
                   (0.0, 1.0, 1.7781, 3.7306),
                   (0.5271, 4.1593, 1.0, 1.3645),
                   (0.8899, 0.2127, 3.4711, 1.0)])

    # print A1
    # print A2
    # print A3

    R1 = np.array([(1.7741), (1.0971), (1.5466), (4.4116)])
    R2 = np.array([(1.0), (0.1358), (1.4936), (4.8486)])
    R3 = np.array([(4.4208), (0.8150), (4.5068), (1.4172)])

    # print R1
    # print R2
    # print R3

    list_of_points = []

    alpha_counter = alpha_min
    while alpha_counter < alpha_max:
        beta_counter = beta_min
        while beta_counter < beta_max:
            new_R = R1 + alpha_counter * (R2 - R1) + beta_counter * (R3 - R1)
            new_A = A1 + alpha_counter * (A2 - A1) + beta_counter * (A3 - A1)
            list_of_points.append((new_R, new_A, alpha_counter, beta_counter))
            beta_counter += (beta_max - beta_min) / beta_steps
        alpha_counter += (alpha_max - alpha_min) / alpha_steps

    return list_of_points
