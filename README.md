# Deep Learning and Behavior Cloning for H2-Diesel Dual-Fuel Engine Control

[![DOI](https://zenodo.org/badge/1036036262.svg)](https://doi.org/10.5281/zenodo.16902579)

This repo contains files related to training the deep learning models for the **H2DF project at the University of Alberta**.  
The project aims to research novel control strategies using the **4.5 L Hydrogen–Diesel Engine at the MECE Engine Lab, Edmonton**.  
This particular repository models the engine combustion process using machine learning, employing various deep learning architectures.

Author: **Alexander Winkler (alexander.winkler@rwth-aachen.de)**

---

## Repository Structure

- **`model_train/`**  
  Contains code to train DNN models for the H₂DF engine using MATLAB Deep Learning Toolbox on experimental datasets, where an expert MPC actuated the engine.  

- **`acados_implementation/`**  
  Contains code to integrate trained models into a behavior cloning (BC) controller. This controller can be:
  - Run in standalone closed-loop simulation with alternative plant models, or  
  - Deployed to embedded hardware (e.g., ESP32, Raspberry Pi) via the acados framework.  

---

## IL Model Training (`model_train/`)

1. Execute `setpath` (drag and drop into MATLAB).  
2. Verify `matlab2tikz` path in `setpath`.  
3. Navigate into the `h2df_model` directory and run scripts from there.  
4. Training will start using the **MATLAB Deep Learning Toolbox**.  
5. Several plotting and evaluation options are available.  

---

## IL Controller (`acados_implementation/`)

1. Execute `setpath` (drag and drop into MATLAB).  
2. Verify correct **acados** installation paths in `env_sh`.  
3. Verify `matlab2tikz` path in `setpath`.  
4. Run `init_simulink` to initialize the Simulink model (uses trained model from `model_train`).  
5. Choose one of the execution modes:
   - **Standalone simulation**: run `sim_xx` scripts.  
   - **Embedded execution**: run `pi_xx` scripts for deployment on Raspberry Pi (ESP32 also possible).  

---

## Dependencies 

The code runs on **MATLAB R2024a or newer** and requires:  
- [Deep Learning Toolbox](https://de.mathworks.com/products/deep-learning.html)  
- [matlab2tikz](https://github.com/matlab2tikz/matlab2tikz) for exporting plots  

---

## Cite us

If you are using this code, please cite the publications:  
- [Dummy1], Paper 1, tbd  
- [Dummy2], Paper 2, tbd
- The data publication on **Zenodo**:
[![DOI](https://zenodo.org/badge/1036036262.svg)](https://doi.org/10.5281/zenodo.16902579)  

---

## License

This project is licensed under the  
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).
