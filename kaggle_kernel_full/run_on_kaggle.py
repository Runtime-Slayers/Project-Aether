import os
import json
import time
from training_pipeline import create_default_config, GNeSAITrainer


# Load config and run training
cfg = create_default_config()
# Use more aggressive training on Kaggle GPU
cfg['num_epochs'] = 100
cfg['batch_size'] = 128
cfg['learning_rate'] = 5e-4
cfg['device'] = 'cuda'
# optional: set seed for reproducibility
cfg['seed'] = 42
cfg['clip_grad_norm'] = 5.0

trainer = GNeSAITrainer(cfg)
results = trainer.run_full_training_pipeline()
print('Training finished. Results:')
print(json.dumps(results, indent=2))

# Compress outputs directory so it's available as kernel output for download
out_dir = trainer.tracker.save_dir
archive_path = os.path.join('outputs', f'run_outputs_{int(time.time())}.tar.gz')
os.makedirs('outputs', exist_ok=True)
import tarfile
with tarfile.open(archive_path, 'w:gz') as tar:
	tar.add(out_dir, arcname=os.path.basename(out_dir))
print(f'Packaged outputs to {archive_path}')

# Optional: if running with internet enabled and kaggle CLI configured in the kernel,
# create a Kaggle Dataset from the outputs so you can download trained artifacts.
if os.environ.get('KAGGLE_CREATE_DATASET', '0') == '1':
	try:
		# Prepare minimal dataset metadata
		ds_title = os.environ.get('KAGGLE_DATASET_TITLE', 'G-NeSAI Outputs')
		ds_id = os.environ.get('KAGGLE_DATASET_ID', None)
		# create dataset metadata file
		meta = {
			'title': ds_title,
			'id': ds_id if ds_id is not None else None,
			'licenses': [{'name': 'CC0-1.0'}]
		}
		meta_path = os.path.join('outputs', 'dataset-metadata.json')
		with open(meta_path, 'w') as f:
			json.dump(meta, f)

		# Call kaggle CLI to create dataset (requires internet & ~/.kaggle/kaggle.json)
		import subprocess
		cmd = ['kaggle', 'datasets', 'create', '-p', 'outputs', '-m', meta_path]
		print('Attempting to create Kaggle Dataset:', ' '.join(cmd))
		subprocess.run(cmd, check=True)
		print('Kaggle Dataset creation attempted (check kernel logs for details).')
	except Exception as e:
		print('Dataset creation skipped or failed:', e)
