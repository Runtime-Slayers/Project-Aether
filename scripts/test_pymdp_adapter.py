import sys
sys.path.insert(0, 'C:/Users/brr33/Downloads/Project Aether')

import torch
from cognitive_core.pymdp_wrapper import PymdpAdapter


def main():
    adapter = PymdpAdapter(n_states=5, device='cpu', lr=1e-2)
    # create dummy batch of observations
    obs = torch.randn(4, 16)  # small flattened observation
    pred = adapter.predict(obs)
    print('predict shape:', pred.shape)
    assert pred.shape == (4, 5)
    # create dummy posterior and run update
    posterior = torch.softmax(torch.randn(4, 5), dim=-1)
    adapter.update(posterior, observation=obs)
    print('update completed')
    print('Test passed')

if __name__ == '__main__':
    main()
