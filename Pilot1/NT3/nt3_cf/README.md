NT3 with counterfactuals:
Code to generate counterfactual examples given an input model and dataset in pkl format. \
Clusters and thresholds counterfactuals, injects noise into dataset \
Workflow: 
1) Generate counterfactuals using cf_nb.py
```
python cf_nb.py
```

2) Create threshold pickle files using threshold.py (provide a threshold value between 0 and 1, see --help) 
```
python threshold.py -d ../nt3.autosave.data.pkl -c cf_redo_all_reformat.pkl -t 0.9 -o threshold_0.9.pkl
```

3) Cluster threshold files using gen_clusters.py
```
python gen_clusters.py -t_value 0.9 -t threshold_0.9.pkl
```

4) Inject noise into dataset using inject_noise.py (provide a scale value to modify the amplitude of the noise, see --help)
```
python inject_noise.py -t threshold_0.9.pkl -c1 cf_class_0_cluster0.pkl -c2 cf_class_1_cluster0.pkl -scale 1.0 -r True -d ../nt3.autosave.data.pkl -f cf_failed_inds.pkl -o noise_data 
```

Abstention with counterfactuals:
Code located in abstention/
Workflow:
1) Run abstention model with nt3_abstention_keras2_cf.py, pass in a pickle file with X (with noise), y (this is the output of 4) above)
2) For a sweep use run_abstention_sweep.sh
3) To collect metrics (abstention, cluster abstention) run make_csv.py
