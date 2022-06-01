set -ex
python train_vqvae_baseline_pytorch.py -e 1
python extract_code_baseline_pytorch.py
python train_pixelsnail_baseline_pytorch.py -e 1
