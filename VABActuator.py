#This is an abstract class which other actuator classes will inherit from
class VABSensor( object ):
    def act(self, value):
        raise NotImplementedError("'act' not yet implemented")
