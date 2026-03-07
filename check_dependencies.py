import importlib
import subprocess
import sys

# List of required Python packages
python_packages = [
    'torch', 'pymdp', 'networkx', 'h5py', 'easyocr', 'matplotlib', 'seaborn', 'jupyter', 'kaggle'
]

# List of CLI tools to check
cli_tools = [
    ['kaggle', '--version']
]

def check_python_packages():
    print('Checking Python packages:')
    for pkg in python_packages:
        try:
            importlib.import_module(pkg)
            print(f'  [OK] {pkg}')
        except ImportError:
            print(f'  [MISSING] {pkg}')

def check_cli_tools():
    print('\nChecking CLI tools:')
    for cmd in cli_tools:
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                print(f'  [OK] {cmd[0]}: {result.stdout.strip()}')
            else:
                print(f'  [MISSING/ERROR] {cmd[0]}: {result.stderr.strip()}')
        except Exception as e:
            print(f'  [MISSING/ERROR] {cmd[0]}: {e}')

if __name__ == '__main__':
    check_python_packages()
    check_cli_tools()
