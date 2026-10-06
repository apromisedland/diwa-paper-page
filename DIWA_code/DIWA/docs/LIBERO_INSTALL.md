# Installation

**(1) Conda Env**
```
conda create -n dreamvla python=3.10
conda activate dreamvla
```

**(2) LIBERO Env**
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install transformers==4.40.2
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install mujoco==2.3.7 robosuite==1.4.0
pip install -e .
```

Python 3.10 with MuJoCo 2.3.7 is the preferred, upstream-compatible setup.
When an existing Python environment has already resolved MuJoCo 3 together
with robosuite 1.4, the LIBERO evaluation, DIWA supervision collector and
smoke-test entry points apply the narrow mass-matrix API compatibility shim
from `utils/libero_compat.py` in memory. The shim does not modify the installed
LIBERO or robosuite package.
