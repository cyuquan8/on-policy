## Installation
```
conda env create -n ENVNAME --file environment.yml
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
git clone -b https://github.com/cyuquan8/on-policy.git
cd on_policy
pip install -e .
git clone -b https://gitlab.com/cmu_aart/asist/gym-dragon
cd gym-dragon
pip install -e .
```