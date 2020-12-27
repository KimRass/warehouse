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
```python
# colab에서 그래프를 그릴 때 한글이 깨지지 않도록 해주는 코드
# 코드 실행 > Colab 상단의 런타임 > 런타임 다시 시작 > 코드 재실행
# 이후에는 matplotlib으로 그리는 그래프에서 한글이 깨지지 않습니다.
import matplotlib.pyplot as plt
import matplotlib as mpl

%config InlineBackend.figure_format = "retina"

!apt -qq -y install fonts-nanum

fpath = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
font = mpl.font_manager.FontProperties(fname=fpath, size=9)
plt.rc("font", family="NanumBarunGothic") 
mpl.font_manager._rebuild()
```
```
function ClickConnect(){
    console.log("코랩 연결 끊김 방지"); 
    document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect, 60 * 1000)
```
