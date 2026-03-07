"""
Lightweight ablation runner for local experiments.
Runs short training jobs with small datasets to produce quick comparative metrics.
"""
import sys
sys.path.insert(0, 'C:/Users/brr33/Downloads/Project Aether')

from training_pipeline import GNeSAITrainer, create_default_config
import json
import random
import numpy as np
import torch


def run_experiment(num_samples=200, num_epochs=1, experiment_name=None):
    cfg = create_default_config()
    cfg['num_epochs'] = num_epochs
    cfg['batch_size'] = 16
    cfg['learning_rate'] = 5e-4
    cfg['seed'] = 42
    if experiment_name:
        cfg['experiment_name'] = experiment_name
    else:
        cfg['experiment_name'] = f'ablation_ns{num_samples}_ne{num_epochs}'

    trainer = GNeSAITrainer(cfg)
    # set seeds for reproducibility inside trainer
    seed = cfg.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    # generate small datasets
    train_data, train_labels = trainer.generate_training_data(num_samples)
    val_data, val_labels = trainer.generate_training_data(max(50, int(num_samples*0.2)))

    # create tiny loaders
    train_loader, val_loader = trainer.create_train_val_loaders  if hasattr(trainer, 'create_train_val_loaders') else (None, None)

    # fallback: run full pipeline but with small sizes
    results = trainer.run_full_training_pipeline()
    print('Experiment', cfg['experiment_name'], 'results:', json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    exps = [ (200,1), (500,1) ]
    all_results = {}
    for ns, ne in exps:
        try:
            r = run_experiment(num_samples=ns, num_epochs=ne)
            all_results[f'ns{ns}_ne{ne}'] = r
        except Exception as e:
            all_results[f'ns{ns}_ne{ne}'] = {'error': str(e)}
    print('All ablation results saved to ablation_results.json')
    with open('ablation_results.json','w') as f:
        json.dump(all_results, f, indent=2)
