# This program takes all voltages and combines them into one big file.

import pandas as pd
import numpy as np
import os
import time

dir_path = "/Users/larserikodegaard/../../Volumes/LE/Mas300/_data/first_30_sek" #Save path

all_files = os.listdir("/Users/larserikodegaard/../../Volumes/LE/Mas300/_data/first_30_sek") #Directory

all_files = all_files[1:] #Takes away the one file in folder that is not voltage

for file in all_files:
    print(file)
all_files = sorted(all_files, key=lambda s: int(s[17:-4])) #Sorts the files in the right order.

# all files is now sorted.

all_voltages = np.zeros([3**13,1201]) #Creates array for all 3**13 somatic potentials.
t0  = time.time()
for i,file in enumerate(all_files):
    print('reading: ',i)
    df = pd.read_csv(dir_path + '/' + file,header=None) #Loads file i
    print('writing: ', i)
    all_voltages[i*10000:(i+1)*10000,:] = df.values #Assembles voltages
    print(time.time() - t0)
    t0 = time.time()

all_voltages_df = pd.DataFrame(all_voltages) #Creates new dataframe for all voltages
print('Saving')

all_voltages_df.to_csv(dir_path + '/' + 'voltages_AP_30sek.csv',header=None,index=None) 
print(time.time() - t0)
print('complete')
