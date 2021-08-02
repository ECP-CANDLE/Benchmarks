./prep.sh
python t29res.py -e 5
python infer.py --model t29res.model.json --weights t29res.model.h5 --n_pred 10
