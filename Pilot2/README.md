## Pilot 2 Benchmarks for Molecular Dynamics Simulation Data

**Overview**: KRAS MD Simulation Data

### Benchmark Specs Requirements

#### Description of the Data
* Data source: MD Simulation output as PDB files (coarse-grained bead simulation)
* Input dimensions:
  * Long term target: ~1.26e6 per time step (6000 lipids x 30 beads per lipid x (position + velocity + type))
  * Current: ~288e3 per time step (6000 lipids x 12 beads per lipid x (position + type))
* Output dimensions: 500
* Latent representation dimension:
* Sample size: O(10^6) for simulation requiring O(10^8) time steps
* Notes on data balance and other issues: unlabeled data with rare events

### Sample Data Sets
* RAS
* 3-component-system (DPPC-DOPC-CHOL)
* af-restraints-290k

#### 3K lipids, 10 microseconds simulation, ~3000 frames:
* Disordered - 3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir
* Ordered - 3k_run32_10us.35fs-DPPC.50-DOPC.10-CHOL.40.dir
* Ordered and gel - 3k_run43_10us.35fs-DPPC.70-DOPC.10-CHOL.20.dir

#### 6K lipids, 25 microseconds simulation, ~7000 frames:

* Disordered - 6k_run10_25us.35fs-DPPC.10-DOPC.70-CHOL.20.dir
* Ordered - 6k_run32_25us.35fs-DPPC.50-DOPC.10-CHOL.40.dir
* Ordered and gel - 6k_run43_25us.35fs-DPPC.70-DOPC.10-CHOL.20.dir

#### Runtime Options
* ```--data-set=3k_Disordered```
* ```--data-set=3k_Ordered```
* ```--data-set=3k_Ordered_and_gel```
* ```--data-set=6k_Disordered```
* ```--data-set=6k_Ordered```
* ```--data-set=6k_Ordered_and_gel```

### Data Set Release Notice

BAASiC Pilot 2 Data Set
Produced at the Lawrence Livermore National Laboratory. 

LLNL-MI-724660
All rights reserved.

This work was performed under the auspices of the U.S. Department of
Energy by Lawrence Livermore National Laboratory under Contract
DE-AC52-07NA27344.

This document was prepared as an account of work sponsored by an
agency of the United States government. Neither the United States
government nor Lawrence Livermore National Security, LLC, nor any of
their employees makes any warranty, expressed or implied, or assumes
any legal liability or responsibility for the accuracy, completeness,
or usefulness of any information, apparatus, product, or process
disclosed, or represents that its use would not infringe privately
owned rights. Reference herein to any specific commercial product,
process, or service by trade name, trademark, manufacturer, or
otherwise does not necessarily constitute or imply its endorsement,
recommendation, or favoring by the United States government or
Lawrence Livermore National Security, LLC. The views and opinions of
authors expressed herein do not necessarily state or reflect those of
the United States government or Lawrence Livermore National Security,
LLC, and shall not be used for advertising or product endorsement
purposes.
