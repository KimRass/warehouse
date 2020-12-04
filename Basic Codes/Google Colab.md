```python
import os, sys
from google.colab import drive
drive.mount("/content/drive")
my_path = "/content/notebooks"
# Colab Notebooks 안에 my_env 폴더에 패키지 저장
os.symlink("/content/drive/My Drive/Colab Notebooks/my_env", my_path)
sys.path.insert(0, my_path)
```
```python
!pip install --target=$my_path [패키지 이름]
```
```python
!git clone https://github.com/kakao/khaiii.git
!pip install cmake
!mkdir build
!cd build && cmake /content/khaiii
!cd /content/build/ && make all
!cd /content/build/ && make resource
!cd /content/build && make install
!cd /content/build && make package_python
!pip install /content/build/package_python
```
