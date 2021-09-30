NT3 with counterfactuals:
Code to generate counterfactual examples given an input model and dataset in pkl format. \
Clusters and thresholds counterfactuals, injects noise into dataset \
Workflow: 
1) Generate counterfactuals using cf_nb.py
2) Create threshold pickle files using threshold.py (provide a threshold value between 0 and 1, see --help) 
3) Cluster threshold files using gen_clusters.py 
4) Inject noise into dataset using inject_noise.py (provide a scale value to modify the amplitude of the noise, see --help)

Abstention with counterfactuals:
Code located in abstention/
Workflow:
1) Run abstention model with nt3_abstention_keras2_cf.py, pass in a pickle file with X (with noise), y (this is the output of 4) above)
2) For a sweep use run_abstention_sweep.sh
3) To collect metrics (abstention, cluster abstention) run make_csv.py