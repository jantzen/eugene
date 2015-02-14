# VABProcess.py

import time
import math

def SameState(state1, state2, tolerance):
    if abs(state1 - state2) < tolerance:
        return True
    else:
        return False


def SymFunc(interface, func, time_var, intervention_var, tolerance, inductive_threshold, time_interval):
    """ Takes a system interface, a function (explicit, no free params)
    representing the intervention being tested for symmetry status, a number
    indicating the variable on which the function is presumed to act, a tolerance (how close
    states need to be in order to count as the same), and an inductive threshold
    (how many affirmative instances are required in order to generalize), and a
    time interval which determines how long to evolve a system and look for
    differences. Returns a logical truth value.
    """

    # EVOLVE AND THEN TRANSFORM
    
    # record the initial state (so it can be replicated in the next step)
    t0 = interface.read_sensor(time_var)
    v0 = interface.read_sensor(intervention_var)
    
    # evolve the system
    time.sleep(time_interval)

    # transform the system
    interface.set_actuator(intervention_var,
            func(interface.read_sensor(intervention_var)))

    # immediately read the new state of affairs
    v1 = interface.read_sensor(intervention_var)

    # TRANSFORM AND THEN EVOLVE

    # get the system back into its initial state and apply the transformation
    # (in a single step)
    interface.set_actuator(intervention_var, func(v0))

    # evolve the system
    time.sleep(time_interval)

    # read the final state of affairs
    v2 = interface.read_sensor(intervention_var)

    
    # COMPARE THE FINAL STATES
    return SameState(v1, v2, tolerance)
