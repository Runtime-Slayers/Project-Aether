import os
import json
from training_pipeline import create_default_config, GNeSAITrainer

# Load config and run training
cfg = create_default_config()
# Use more aggressive training on Kaggle GPU
cfg['num_epochs'] = 100
cfg['batch_size'] = 128
cfg['learning_rate'] = 5e-4
cfg['device'] = 'cuda'
# optional: set seed for reproducibility
cfg['seed'] = 1
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
