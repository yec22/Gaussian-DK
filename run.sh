# Train with train/test split
python train.py -r 4 -s datasets/dark/piano -m output/piano --port 1111 --eval

# Generate renderings
python render.py -m output/piano

# Compute metrics on renderings
python metrics.py -m output/piano

# More visualization
python render_spherify.py -m output/piano