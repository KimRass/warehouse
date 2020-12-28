# Google Drive에 Mount하기
```python
from google.colab import drive
    drive.mount("/content/drive", force_remount=True)
import os
import sys
try:
    my_path = "/content/notebooks"
    os.symlink("/content/drive/MyDrive/ColabNotebooks/my_env", my_path)
    sys.path.insert(0, my_path)
except:
    pass
cd /content/notebooks
```
# Package 영구 설치
```python
!pip install --target=$my_path [패키지 이름]
```
# Install khaiii
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
# 한글 출력
```python
import matplotlib as mpl
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = "retina"
!apt -qq -y install fonts-nanum
fpath = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
font = mpl.font_manager.FontProperties(fname=fpath, size=9)
plt.rc("font", family="NanumBarunGothic") 
mpl.font_manager._rebuild()
mpl.rcParams["axes.unicode_minus"] = False
```
# 연결 끊김 방지
```
function ClickConnect(){
    console.log("코랩 연결 끊김 방지"); 
    document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect, 60 * 1000)
```
