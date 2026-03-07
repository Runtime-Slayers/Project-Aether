import sys
sys.path.insert(0, 'C:/Users/brr33/Downloads/Project Aether')

try:
    from cognitive_core import active_inference
    print('import_active_inference_ok')
    from cognitive_core import pymdp_wrapper
    print('import_pymdp_wrapper_ok')
except Exception as e:
    print('IMPORT_ERROR', repr(e))
