# conda cheatsheet
conda create --name py38 --clone py36

# 通过list迁移
conda list --explicit > spec-list.txt
conda create  --name python-course --file spec-list.txt

# 通过环境包迁移
conda env export > requirements.yml
conda env create -f environment.yml

# 通过conda-pack迁移
conda install -c conda-forge conda-pack
pip install conda-pack
conda pack -n my_env
conda pack -n my_env -o out_name.tar.gz
conda pack -p /explicit/path/to/my_env

mkdir -p my_env
tar -xzf my_env.tar.gz -C my_env

./my_env/bin/python

source my_env/bin/activate

(my_env) $ conda-unpack

# https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-from-file
# https://conda.github.io/conda-pack/
