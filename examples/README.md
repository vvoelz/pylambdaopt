# Example pylambdaopt calculations

## Alchemical free energy of mutation A16F in miniprotein A8

<img width="331" alt="image" src="https://github.com/user-attachments/assets/31352f76-8197-4afa-bf2f-fdff4e8bc277" />

Uses script: `../script/optimize_FEP_lambdas.py`.  Input data in `/FEP_A8_A16F_unopt`

* `./runme_FEP_A8_A16F` will optimize spacing
* `./runme_FEP_A8_A16F_optimize_K` will optimize spacing and number 

#### Reference
Massively Parallel Free Energy Calculations for in silico Affinity Maturation of Designed Miniproteins.
Dylan Novack, Si Zhang, Vincent A. Voelz. _BioRxiv_
doi: https://doi.org/10.1101/2024.05.17.594758



## Absolute free energy of decoupling Sulfamethazine from water

<img width="100" alt="image" src="https://github.com/user-attachments/assets/d4d26402-a476-465e-b169-a23cbb07a023" />

Uses script: `../script/optimize_coul_vdw_lambdas.py`.  Input data in `/Sulfamethazine_tau0_0_water`

* `./runme_Sulfamethazine_tau0_0_water` will optimize spacing
* `./runme_Sulfamethazine_tau0_0_water_optimize_K` will optimize spacing and number

#### Reference
Expanded Ensemble Predictions of  Toluene--Water Partition Coefficients in the SAMPL9 LogP Challenge.
Goold, Steven, Robert M. Raddi, and Vincent Voelz. _ChemRxiv_, 2024
doi: https://doi.org/10.26434/chemrxiv-2024-rfkkp. 



## Absolute free energy of decoupling donepezil from its bound complex with Acetylcholine esterase (AChE)

<img width="599" alt="image" src="https://github.com/user-attachments/assets/ba3bd33e-8e79-4234-acb1-d488cf15766b" />

Uses script: `../script/optimize_rest_coul_vdw_lambdas.py`.  Input data in `/donepezil_rest_coul_vdW`

* `./runme_donepezil_rest_coul_vdW` will optimize spacing
* `./runme_donepezil_rest_coul_vdW_optimize_K` will optimize spacing and number



