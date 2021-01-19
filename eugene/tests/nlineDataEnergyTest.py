from energy_test_stat import twoSample

a = '../src/auxiliary/durations_70_removed_sample.csv'
b = '../src/auxiliary/durations_no_filter.csv'

def convertio(line):
    #print(line)
    try:
        q = float(line)
        return q 
    except:
        print('this is not a number', line)

with open(a, 'r') as thefile:
    #a = thefile.read().split('\n')
    #a_new = [y for y in a if y != '']
    someFewer = [convertio(x) for x in thefile.read().split('\n') if x != '']

with open(b, 'r') as thefile:
    noFewer = [convertio(x) for x in thefile.read().split('\n') if x != '']

print("someFewer vs noFewer", twoSample(someFewer,noFewer))
print("someFewer vs someFewer", twoSample(someFewer, someFewer))
print("noFewer vs noFewer", twoSample(noFewer, noFewer))
