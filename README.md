# Installation

```bash
conda env create -n ENVNAME --file environment.yml
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
git clone -b https://github.com/cyuquan8/on-policy.git
cd on_policy
pip install -e .
git clone -b https://gitlab.com/cmu_aart/asist/gym-dragon
cd gym-dragon
pip install -e .
```
