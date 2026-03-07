Kaggle GPU run instructions

This project includes a helper to push and run the full `G-NeSAI` training pipeline on Kaggle using the Kaggle CLI.

Prerequisites (on your laptop):
- Python 3.8+ and `kaggle` CLI installed (`pip install kaggle`).
- Your Kaggle API credentials file (JSON), e.g. `kaggle_rampanther.json`.
- Ensure the project repo is the working directory when running the helper.

Steps to run on Kaggle:

1. Copy your credentials (one-time):

```bash
# from Windows PowerShell
mkdir -p $env:USERPROFILE/.kaggle
cp C:\Users\brr33\Downloads\kaggle_rampanther.json $env:USERPROFILE/.kaggle/kaggle.json
# on Windows ensure correct permissions (Git Bash or WSL recommended):
chmod 600 $env:USERPROFILE/.kaggle/kaggle.json
```

2. Push and run the kernel using the helper script (from project root):

```bash
# make the helper executable (Git Bash / WSL)
chmod +x tools/run_on_kaggle.sh
# then run, providing path to your kaggle json
tools/run_on_kaggle.sh C:/Users/brr33/Downloads/kaggle_rampanther.json
```

Notes:
- The helper creates a kernel directory `kaggle_kernel_full/` and pushes a script kernel that enables GPU.
- Kaggle may queue the job; monitor the run on your Kaggle account under "My Kernels".
- Internet is disabled in the kernel metadata for security; if you need internet, set `enable_internet` to true in `kaggle_kernel_full/kernel-metadata.json`.

If you want, I can also prepare a Kaggle Dataset upload and a kernel that mounts it (useful for large generated datasets).