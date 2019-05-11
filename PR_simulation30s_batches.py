import neuron
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

### Load files
neuron.h.load_file("mosinit.hoc") #Loads the model downloaded from modelDB.


### Setters ###

def set_cm(x):

    neuron.h.rinzelnrn[0].soma(0.5).cm = x
    neuron.h.rinzelnrn[0].dend(0.5).cm = x

def set_gl(x):

    neuron.h.rinzelnrn[0].soma(0.5).pas.g = x
    neuron.h.rinzelnrn[0].dend(0.5).pas.g = x

def set_gna(x):

    neuron.h.rinzelnrn[0].soma(0.5).nafPR.gmax = x

def set_el(x):

    neuron.h.rinzelnrn[0].dend(0.5).pas.e = x
    neuron.h.rinzelnrn[0].soma(0.5).pas.e = x

def set_ena(x):

    neuron.h.erev_nafPR = x

def set_ek(x):

    neuron.h.erev_kdr = x

def set_gc(x):

    neuron.h.gc = x

    for seg in neuron.h.allsec():
        seg.Ra = 1

    ga = (1e-6/(neuron.h.gc/neuron.h.pp * (neuron.h.area(0.5)*1e-8) * 1e-3))/(2*neuron.h.ri(0.5))

    for seg in neuron.h.allsec():
        seg.Ra = ga

def set_gdr(x):

    neuron.h.rinzelnrn[0].soma(0.5).kdr.gmax = x

def set_gahp(x):

    neuron.h.rinzelnrn[0].dend(0.5).rkq.gmax = x

def set_gC(x):
    neuron.h.rinzelnrn[0].dend(0.5).kcRT03.gmax = x

def set_gca(x):
    
    neuron.h.rinzelnrn[0].dend(0.5).cal.gmax = x

def set_eca(x):

    neuron.h.erev_cal = x

def set_is(x):
    neuron.h.stim[0].amp = x

def set_p(x):
    
    neuron.h.pp = x

    for seg in neuron.h.allsec():
        seg.Ra = 1

    ga = (1e-6/(neuron.h.gc/neuron.h.pp * (neuron.h.area(0.5)*1e-8) * 1e-3))/(2*neuron.h.ri(0.5))

    for seg in neuron.h.allsec():
        seg.Ra = ga

def set_params(param):

    set_cm(param[0])

    set_gl(param[1])

    set_gna(param[2])

    set_el(param[3])

    set_ena(param[4])

    set_ek(param[5])

    set_gc(param[6])

    set_gdr(param[7])

    set_gahp(param[8])

    set_gC(param[9])

    set_gca(param[10])

    set_eca(param[11])

    set_p(param[12])

    

### Simulation ###
def simulation():
    
    neuron.h.tstop = 40 #Sets Simulation time
    somav = neuron.h.Vector() #Creates a voltage recorder.
    t = neuron.h.Vector() #Creates a time recorder.

    somav.record(neuron.h.rinzelnrn[0].soma(0.5)._ref_v) #Records the somatic membrane potential
    t.record(neuron.h._ref_t) #Initiates the time recorder

    neuron.h.finitialize()
    neuron.h.run(40) #Runs simulation for 40 seconds.

    time = copy.copy(t.as_numpy()) #Create a copy of the recorded time
    voltage = copy.copy(somav.as_numpy()) # Creates a copy of the recorded somatic voltage

    f = interp1d(time,voltage)

    new_time =  np.linspace(0,30,1201) #Creates a new time-array for 30 seconds
    new_voltage = f(new_time) #Using interpolation to calculate all timepoints for the 30 seconds.
    


    return new_voltage,new_time


t0 = time.time() # Measures the time usage
parameter_combinations = np.loadtxt("/Users/larserikodegaard/../../Volumes/LE/MAS300/_data/para_comb_20p.csv",delimiter=",") # Loads the parameter combinations generated with PR_samplespace_generator.py


print(time.time() - t0) # Prints time used for loading
print('Param_combinations is imported: ',parameter_combinations.shape)


t0 = time.time()
for j in range(0,int(1590000/10000 + 1)): #Simulates in batches of 10 000. (Time consuming)

    voltages = np.zeros([10000,1201]) # Empty array for somatic potential


    for i,params in enumerate(parameter_combinations[10000*j:10000*(j+1),:]):

        set_params(params)
        v,t_v = simulation()
        voltages[i,:] = v
        if i%1000 == 0:
            print(str(j*10000 + i)+':',time.time() - t0)
            t0  = time.time()
# Saves the simulated batch.
    np.savetxt("/Users/larserikodegaard/../../Volumes/LE/Mas300/_data/first_30_sek/voltages_AP_30sek"+str(j)+".csv",voltages, delimiter=",")

    print('save time:' ,time.time()-t0)
    print(j)
