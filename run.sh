CUDA_VISIBLE_DEVICES=0 python train.py -r 4 -s datasets/dark/piano -m output/ablation/wo-grad-scale/piano --port 1111 --eval  # Train with train/test split
CUDA_VISIBLE_DEVICES=0 python render.py -m output/handcraft/piano # Generate renderings
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/ablation/wo-grad-scale/piano # Compute error metrics on renderings
CUDA_VISIBLE_DEVICES=0 python render_spherify.py -m output/handcraft/piano