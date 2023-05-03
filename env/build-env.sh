conda env create -f ml4cfd-pytorch8-env.yml
conda install -n  ml4cfd-pytorch8-env pytorch==1.8.1 torchvision==1.8.1 torchaudio==1.8.1 cudatoolkit=10.2 -c pytorch
conda install -n  ml4cfd-pytorch8-env pyg -c pytorch -c pyg -c conda-forge
conda install -n  ml4cfd-pytorch8-env pytorch-lightning -c conda-forge

conda env create -f ml4cfd-pytorch9-env.yml
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
conda install pyg -c pytorch -c pyg -c conda-forge
conda install pytorch-lightning -c conda-forge

conda env create -f psignn-pytorch9-env.yml
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
conda install pyg -c pytorch -c pyg -c conda-forge
conda install pytorch-lightning -c conda-forge
pip install setuptools==59.5.0
