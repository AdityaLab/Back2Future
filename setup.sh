source ~/anaconda3/bin/activate
conda env create --prefix ./envs/backfill --file environment.yml
conda activate ./envs/backfill
conda clean --all
