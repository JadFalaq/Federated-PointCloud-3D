$ErrorActionPreference = "Stop"

if (-not (Test-Path .venv)) {
    python -c "import venv; venv.EnvBuilder(with_pip=True).create(r'.venv')"
}

$python = Resolve-Path ".\.venv\Scripts\python.exe"
& $python -m ensurepip --upgrade
& $python -m pip install --upgrade --ignore-installed pip setuptools wheel
& $python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
& $python -m pip install torch-geometric==2.6.1 plyfile==1.0.3 pandas==2.0.3 scikit-learn==1.3.2 matplotlib==3.7.5 seaborn==0.13.2 PyYAML==6.0.1 tqdm==4.66.4 numpy==1.24.4
