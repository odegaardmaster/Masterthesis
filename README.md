# Metamodelling of the Hodgkin-Huxley Model and the Pinsky-Rinzel Model Using Local Multivariate Regression and Deep Learning

This repository is created for my master thesis, perfomered as a part of the  Master  program  in Data Science, at the Faculty of Science and Technology (REALTEK) at the Norwegian University of Life Sciences (NMBU) the spring 2019.


## Data Generation
### the Hodgkin-Huxley model

The parameter space and membrane potential used for metamodelling of Hodgkin-Huxley model are generated in:
  - HH_datagenerator.py
  
### the Pinsky-Rinzel model
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

- HH_classical.ipynb (Classical metamodelling)
- HH_classical_aggregated.ipynb (Classical metamodelling of aggregated phenotypes)
- HH_inverse.ipynb (Inverse metamodelling)

### Models used in metamodelling of Pinsky-Rinzel model
  
- PR_classical.ipynb (Classical metamodelling)
- PR_classical_aggregated.ipynb (Classical metamodelling of aggregated phenotypes)
- PR_inverse.ipynb (Inverse metamodelling)


## HCPLSR 

### HCPLSR implimentation

The matlab implementation of HCPLSR used for metamodelling:
- HPLSR.m (Calibrating models)

- HPLSRpred.m (Prediction model)

- Example_HCPLSR.m (Example run of HCPLSR metamodelling and saving structures for plotting)

### HCPLSR plots

- HCPLSR_HH_plots.ipynb (Prediction results achived with HCPLSR metamodelling of the Hodgkin-Huxley model)
- HCPLSR_PR_plots.ipynb (Prediction results achived with HCPLSR metamodelling of the Pinsky-Rinzel model)

### Sensitivity Analaysis

- HHagmatlabplotting.ipynb (Sensitivity analysis classical metamodelling of aggregated features extracted from the Hodgkin-Huxley model)
- PRagmatlabplotting.ipynb (Sensitivity analysis classical metamodelling of aggregated features extracted from the Pinsky-Rinzel model)
- HHmatlabplotting.ipynb (Sensitivity analysis classical metamodelling of Hodgkin-Huxley model)
- PRmatlabplotting.ipynb (Sensitivity analysis classical metamodelling of Pinsky-Rinzel model)


