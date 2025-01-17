# Install
- Reference: https://ttottoro.tistory.com/387
- `ls -l /usr/local/bin/python*`을 실행했을 때 출력되는 게 있다면 None-Homebrew 파이썬이 설치되어 있다는 것입니다. 왜냐하면 경로다 ".../Library/Frameworks/Python.framework"이기 때문입니다.
- "/usr/bin/python"은 절대 삭제하면 안 된다고 합니다.
- Homebrew를 통해서 설치한다면 경로는 ".../Cellar/python"이 되기 때문입니다.
```sh
export PATH=/usr/local/bin:/usr/local/sbin:${PATH}
export PATH=${PATH}:/Users/jongbeom.kim/Library/Python/3.8.bin

brew install python3
```

# Python Version
```sh
python --version
# Or
python -V
```
# List Python Versions
```sh
# Brew를 통해서 설치된 Python은 "/usr/local/bin/python*"에 설치됩니다.
ls -l /usr/local/bin/python*
```
# Change Default Python Version
```sh
# Example (Python3.6)
ln -s -f /usr/local/bin/python3.6 /usr/local/bin/python3
```

# Bash Profile
```sh
# Open
open ~/.bash_profile

# source ~/.bash_profile
```
```sh
export PYTHONPATH="/Users/jongbeom.kim/Desktop/workspace/flitto/b2btools/apps"
export PYTHONPATH="/Users/jongbeom.kim/Desktop/workspace/flitto/data_mgmt/apps"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/jongbeom.kim/Desktop/workspace/Github/Work/flitto-351906-36a591c7de9c.json"
export PYENV_ROOT="$HOME/.pyenv"
# Define environment variable `PYENV_ROOT` to point to the path where Pyenv will store its data. `"$HOME/.pyenv"` is the default. If you installed Pyenv via Git checkout, we recommend to set it to the same location as where you cloned it.
export PATH="$PYENV_ROOT/bin:$PATH"
# Run eval `"$(pyenv init -)"` to install `pyenv` into your shell as a shell function, enable shims and autocompletion
eval "$(pyenv init -)"
# pyenv-virtualenv: prompt changing will be removed from future release. configure `export PYENV_VIRTUALENV_DISABLE_PROMPT=1` to simulate the behavior.
export PYENV_VIRTUALENV_DISABLE_PROMPT=1
```

# `venv`
```sh
python -m venv <virtual_environment_name>

# Activate
# Windows
cd <virtual_environment_name>/Scripts
activate
# MacOS
source <virtual_environment_name>/bin/activate
```

# `pyenv`
## Install
```sh
brew install pyenv
brew install pyenv-virtualenv
```
```sh
# List All Installable Python Versions
pyenv install --list

# List All Installed Python Versions
pyenv versions
```
```sh
# Install Python
pyenv install <version>

# Uninstall Python
pyenv uninstall <version>
```
## Virtual Environments
```sh
# Create Virtual Environment
pyenv virtualenv [<python_version>] [<virtual_environment_name>]

# Activate virtual environment
pyenv activate <virtual_environment_name>

# Deactivate virtual environment
pyenv deactivate <virtual_environment_name>

# Delete Virtual Environment
pyenv uninstall <virtual_environment_name>

# List virtual environments
pyenv virtualenvs
```
## In Virtual Envionment
```sh
# Locate Python Interpreter
pyenv which python

# Check Python version
python -V
# Or
python --version
```
## 해당 Python 버전을 기본으로 설정
```sh
pyenv global 3.9.0
```

# pip (Package Installer for Python)
## Upgrade pip
```sh
pip install --upgrade pip
```

# Kill
```sh
ps -ef | grep -i python
kill -9 ...
```

# C++ 관련
```sh
pip install cython
brew install gcc
xcode-select --install
```

# GPU
- References:
    - https://velog.io/@ksy5098/CUDA-cuDNN-%EC%84%A4%EC%B9%98-%EB%B2%84%EC%A0%84-%ED%99%95%EC%9D%B8
## 1) CUDA version
```sh
nvcc -V
```
```python3
torch.version.cuda
```
# cuDNN version
```sh
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```
```python3
torch.backends.cudnn.version()
```

# TensorRT
```sh
pip install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases
```
