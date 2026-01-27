rm -Rf .venv/

module purge
module load python/3.11 uv
uv venv --system-site-packages
source .venv/bin/activate
uv pip install numpy ipython matplotlib scikit-learn
