#This is an abstract class which other sensor classes will inherit from
class VABSensor( object ):
    def read(self):
        raise NotImplementedError("'read' not yet implemented")