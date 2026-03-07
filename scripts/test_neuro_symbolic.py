import sys
sys.path.insert(0, 'C:/Users/brr33/Downloads/Project Aether')

from neuro_symbolic.interface import NeuroSymbolicInterface
from cognitive_core.symbolic_rules import create_sample_context


def main():
    ctx = create_sample_context()
    iface = NeuroSymbolicInterface(n_classes=6)
    mask = iface.get_allowed_mask(ctx)
    print('Allowed mask:', mask.tolist())
    assert mask.sum() > 0, 'No allowed actions — check rules/context'
    print('Test passed')

if __name__ == '__main__':
    main()
