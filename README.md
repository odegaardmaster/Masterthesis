# Masterthesis
This repository is created for my master theses.


## Data Generation
### Hodgkin-Huxley

The parameter space and membrane potential used for metamodelling of Hodgkin-Huxley model are generated in:
  - HH_datagenerator.py
  
### Pinsky-Rinzel
The parameter space used in metamodelling of Hodgkin-Huxley model is generated in:
  - PR_samplespace_generator.py
  
The somatic voltages are simulated first in batches, then assembled:
  - PR_simulation30s_batches.py (batches)
  
  - PR_simulation_assamble.py (assembler)
  
### Aggregated Features
The agregated features was calculated in:
- Aggregated_features.ipynb (Aggregated features and other membrane potential behaviours.)
  
## Deep learning

### Models used in metamodelling of Hodgkin-Huxley model

- HH_classical.ipynb
- HH_classical_aggregated.ipynb
- HH_inverse.ipynb

### Models used in metamodelling of Pinsky-Rinzel model
  
- PR_classical.ipynb
- PR_classical_aggregated.ipynb
- PR_inverse.ipynb


## HCPLSR 

### HCPLSR implimentation

The matlab implementation of HCPLSR used for metamodelling:
- HPLSR.m (Calibrating models)

- HPLSRpred.m (Testing model)

### HCPLSR plots

- HCPLSR_HH_plots.ipynb (Prediction results achived with HCPLSR metamodelling of the Hodgkin-Huxley model)
- HCPLSR_PR_plots.ipynb (Prediction results achived with HCPLSR metamodelling of the Pinsky-Rinzel model)

### Sensitivity Analaysis

- HHagmatlabplotting.ipynb (Sensitivity analysis classical metamodelling of aggregated features extracted from the Hodgkin-Huxley model)
- PRagmatlabplotting.ipynb (Sensitivity analysis classical metamodelling of aggregated features extracted from the Pinsky-Rinzel model)
- HHmatlabplotting.ipynb (Sensitivity analysis classical metamodelling of Hodgkin-Huxley model
- PRmatlabplotting.ipynb (Sensitivity analysis classical metamodelling of Pinsky-Rinzel model


