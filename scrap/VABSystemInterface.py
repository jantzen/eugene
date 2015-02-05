#Interface for dealing with the (virtual or simulated) sensors and actuators. 
class VABSystem( object ):
    def __init__(self, sensors, actuators):
        _sensors = sensors;
        _actuators = actuators;
        _sensorData = [];
    
    def read(self):
        for sensor in self._sensors:
            self._sensorData.add(sensor.read())
        #Then store the data from sensorData into the database
        
    def intervene(self, actuatorID, value):
        self._actuators[actuatorID].act(value)