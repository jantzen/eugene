"""robustness checks

e.g., 

AGWNRobustness

Simplest, but the idea would just be to add increasing levels of Gaussian noise 

If you were to model off testClassifier in concentration example, with:
    
xsensor = VABCircuitSensor([dynamic range], noise_stdev, etc.)

you increment the value in noise_stdev

check accuracy of classifications, repeat, returning performance at different SNR's

"""
