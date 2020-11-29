# Ubuntu
## install
```
sudo apt-get upgrade
```
```
sudo atp-get update
```
```
sudo apt-get install python3-pip
```
```
sudo pip3 install jupyter numpy pandas
```
## install
### cmake
```
sudo pip3 install cmake
```
### cython
```
sudo pip3 install cython
```
### n2
```
sudo pip3 install n2
```
### khaii
#### in Google Colab
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
#### in Windows 10
```
git clone https://github.com/kakao/khaiii.git
```
- 깃헙에서 khaiii 관련 파일을 복사한다.
```
cd khaiii
```
- khaiii 폴더로 이동(cd = change directory)
```
mkdir build
```
- khaiii 폴더 아래에 build 폴더 만들기(mkdir = make directiory)
```
cd build
```
- build 폴더로 이동
```
sudo cmake ..
```
- 프로그램에 필요한 리소스 준비(끝에 점 2개 찍어야 함) - 약 10분 정도 걸림
```
sudo make all
```
- 빌드 실행 - 약 5분 걸림
```
sudo make large_resource # large
sudo make resource       # base
```
- 리소스 빌드(large와 base중 선택, 저장용량에 문제가 없다면 large 선택)
```
sudo make install 
```
- khaiii 설치
```
sudo make package_python
cd package_python
```
- python과 바인딩
```
sudo pip3 install .
```
- 마지막에 점 하나 찍어야 함, 약 5분 걸림
## jupyter notebook
```
jupyter notebook --allow-root --no-browser
```
```
http://127.0.0.1:8888/?token=f1ee03e6b6142b567d0d00076e6a11618d18561d7905ddea
```
