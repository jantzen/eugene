
import time
import math
import random
import numpy as np
import pdb
import scipy.stats as stats


# import interface
import eugene as eu

#Classes:
# Category

#Functions:
# Classify 


####################################################################
####################################################################
####################################################################
#Classes:
#
#  Catgetory
#
#

class Category( object ):
    """ Holds a list of system ids belonging to the same dynamical kind.
    """
    def __init__(self, systems=set([])):
        self._systems = systems
#        self._paradigm = paradigm

    def add_system(self, sys):
        self._systems = self._systems.union(set([sys]))

#    def update_paradigm(self, new_paradigm_system):
#        self._paradigm = new_paradigm_system


####################################################################
####################################################################
####################################################################
#Functions:

def Classify(system_ids, models):
    """ Assumes that the ith model corresponds to sys_id i.
    """
#    pdb.set_trace()
    # initialize the sort with the first system in the list of systems
    classes = []
    classes.append(Category(set([system_ids[0]])))

    # sort the remainder of the systems
    for sys_id in system_ids[1:]:
        categorized = False
        for c in classes:
            # compare the unknown system to each system in c
            same_category = True
            for system in c._systems:
                result = compare.CompareModels(models[sys_id], models[system])
                if result == 1:
                    same_category = False
                    print('{0} is different from {1}'.format(sys_id, system))
                    break
            if same_category:
                c.add_system(sys_id)
                categorized = True
                print('{0} is the same as {1}'.format(sys_id, c._systems))
                break

        # if the system doesn't fit a known category, make a new one
        if categorized == False:
            classes.append(Category(set([sys_id])))

#    # go back and try to classify any singletons
#    revised_classes = []
#    singletons = []
#    for c in classes:
#        if len(c._systems) == 1:
#            singletons.append(c)
#        else:
#            revised_classes.append(c)
#    for s in singletons:
#        sys_id = s._systems.pop()
#        categorized = False
#        for c in revised_classes:
#            # compare the unknown system to the paradigm
#            result = compare.CompareModels(models[sys_id], c._paradigm)
#            if result != None:
#                categorized = True
#                c.add_system(sys_id)
#                c.update_paradigm(result)
#                break
#        # if the system doesn't fit a known category, make a new one
#        if categorized == False:
#            revised_classes.append(Category(set([sys_id]), models[sys_id]))
#
#    classes = revised_classes

    # return the list of classes
    return classes

