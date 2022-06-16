# Python Version
```sh
python --version
# Or
python -V
```
# List Python Versions
- Brew를 통해서 설치된 Python은 "/usr/local/bin/python*"에 설치됩니다.
```sh
ls -l /usr/local/bin/python*
```
# Change Default Python Version
```sh
# Example (Python3.6)
ln -s -f /usr/local/bin/python3.6 /usr/local/bin/python3
```

# `pyenv`
## 설치 가능한 Python 버전
```sh
pyenv install --list
```
## 특정한 버전 Python 삭제
```sh
pyenv uninstall 3.9.0
```
## 설치된 Python list
```sh
pyenv versions
```
## 해당 Python 버전을 기본으로 설정
```sh
pyenv global 3.9.0
```
## Create Virtual Environment
```sh
pyenv virtualenv [version] [name]
```
## 가상환경 시작하기
```sh
pyenv activate py39
```
## 가상환경 종료하기
```sj
pyenv deactivate
```
## 가상환경 목룍보기
```sh
pyenv virtualenvs
```

# pip (Package Installer for Python)
## Upgrade pip
```sh
python -m pip install --upgrade pip
```
