# SystemClass.py

class System(object):
    """ Class to represent a system of variables on which automated discovery
    routines can act. Systems may be real, physical systems consisting of a set
    of sensors and actuators, or simulated.

    Programing note: to use decorators like "@property", you must inherit the
    object class.
    """
    def __init__(self, n=2):
        self._num_vars = n

    def __str__(self):
        return "System with %s variables" % (self._num_vars)

    def __repr__(self):
        return str(self)

    # setters and getters for the single class property

    @property 
    def n(self):
        """ Return the number of variables in the system.
        """
        return self._num_vars
   
    @n.setter
    def n(self,n):
        if n >= 2:
            self._num_vars = n

    """ placeholders for polymorphic functions that will be instantiated in each
    child class.
    """
    # def read(self, var)
    # def set(self, var)
