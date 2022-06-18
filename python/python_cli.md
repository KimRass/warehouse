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
pyenv virtualenv [version] [name]

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
