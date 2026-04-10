ml purge
setup_accre_software_stack
ml load python/3.11.5 cuda/12.6
source /home/medani/clam_boss_env/bin/activate
export UV_INDEX_STRATEGY="unsafe-best-match"
export UV_CONSTRAINT="/cvmfs/soft.computecanada.ca/config/python/constraints.txt"
uv sync --active
