# Masterthesis
This repository is created for my master theses.

# Hodgkin-Huxley

The parameter space and membrane potential used for metamodelling of Hodgkin-Huxley model are generated in:
  - HH_datagenerator.py
  
# Pinsky-Rinzel
The parameter space used in metamodelling of Hodgkin-Huxley model is generated in:
  - PR_samplespace_generator.py
  
The somatic voltages are simulated first in batches, then assembled:
  - PR_simulation30s_batches.py (batches)
  
  - PR_simulation_assamble.py (assembler)
  
