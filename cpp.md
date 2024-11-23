# Settings
- https://velog.io/@progdevelog/%EB%A7%A5%EC%97%90%EC%84%9C-CC-%EC%96%B8%EC%96%B4-%EC%BD%94%EB%94%A9%ED%95%98%EA%B8%B0
- Code Runner 설치.
- 'Run In Terminal' 설정.
- 'Run Code'로 코드 실행 ('fn' + 'option').

# CMake

## Install
```bash
# MacOS
brew install cmake
```

## 'CMakeLists.txt'
```txt
# CMake의 최소 버전 설정
cmake_minimum_required(VERSION <MINIMUM_CMAKE_VERSION>)

# 프로젝트 이름과 버전 설정
project(
    <PRJ_NAME>
    VERSION <PRJ_VERSION>
)

# 소스 파일 지정
set(SOURCES main.cpp otherfile.cpp)
# set은 CMake에서 사용할 변수를 설정하는 명령어이다. 여기서는 C++ 표준을 결정하는 CMAKE_CXX_STANDARD를 20으로 설정해주었다. 이렇게 하면 C++ 표준을 C++20로 사용하겠다는 의미이다.
set(CMAKE_CXX_STANDARD 20)

# 실행 파일 생성
add_executable(<EXECUTABLE_NAME> <SCRIPT_FILE_TO_BUILD1>, ...)

# 루트 디렉토리에 있는 CMakeLists.txt 파일에는 서브 디렉토리를 지정해주어야 한다.
add_subdirectory(<SUBDIR)

# 외부 라이브러리 링크 예시
# `REQUIRED`: 필수.
# find_package(<PKG_NAME> REQUIRED)
# target_link_libraries(<EXECUTABLE_NAME>
    Boost::Boost
    ...
)
```
```txt
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "*.cpp")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
```

## Run CMake
```bash
[mkdir build]
[cd build]
cmake ..
```

## Build (Compile?) Project
```bash
make
```

## Run Executable
```
./EXECUTABLE
```

# LibTorch
- Reference: https://velog.io/@dev_junseok_22/C%ED%99%98%EA%B2%BD%EC%97%90%EC%84%9C-TorchScript-%EB%AA%A8%EB%8D%B8-%EB%A1%9C%EB%93%9C
```bash
mkdir build
cd build
# `..`: 'CMakeLists.txt'의 상위 디렉토리.
# 완료 시: "-- Build files have been written to: ..."
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
# `Release`: 최적화가 적용된 상태로 빌드되어, 개발 및 테스트가 아닌 실제 배포에 적합한 실행 파일을 생성합니다.
# 완료 시: "[100%] Built target main"
cmake --build . --config Release
```

# OpenCV
- References:
    - https://dudgus907.tistory.com/73
    - https://blog.minsulee.com/145
```bash
[brew install opencv]
[brew install pkg-config]
[pkg-config --modversion opencv4]  # Verify OpenCV Installation.
brew info opencv
# Check 'include' directory, e.g., '/opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4'.
    # `ls /opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4`: You should see OpenCV headers like 'opencv2'.
# Check 'lib' directory, e.g., '/opt/homebrew/Cellar/opencv/4.10.0_12/lib'.
    # `ls /opt/homebrew/Cellar/opencv/4.10.0_12/lib`: You should see '.dylib' files like: 'libopencv_core.dylib', 'libopencv_imgcodecs.dylib', 'libopencv_highgui.dylib'.
# Set 'CMakeLists.txt', e.g.,:
    # cmake_minimum_required(VERSION 3.10)
    # project(OpenCV_Project)

    # # Set C++ Standard
    # set(CMAKE_CXX_STANDARD 14)

    # # Manually specify OpenCV paths
    # set(OpenCV_INCLUDE_DIRS /opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4)
    # set(OpenCV_LIBS 
    #     /opt/homebrew/Cellar/opencv/4.10.0_12/lib/libopencv_core.dylib
    #     /opt/homebrew/Cellar/opencv/4.10.0_12/lib/libopencv_imgcodecs.dylib
    #     /opt/homebrew/Cellar/opencv/4.10.0_12/lib/libopencv_highgui.dylib
    #     # Add other libraries as needed
    # )

    # # Include OpenCV headers
    # include_directories(${OpenCV_INCLUDE_DIRS})

    # # Add Executable
    # add_executable(opencv opencv.cc)

    # # Link OpenCV libraries
    # target_link_libraries(opencv ${OpenCV_LIBS})
```
<!-- ```bash
# 'C/C++: Edit Configurations (UI)' (= 'c_cpp_properties.json') -> 'Include path': Add the compiler path, e.g., '"/opt/homebrew/Cellar/opencv/4.10.0_12/include/**'
# 'tasks.json':
    # "`pkg-config",  // OpenCV
    # "opencv4",  // OpenCV
    # "--libs",  // OpenCV
    # "--cflags",  // OpenCV
    # "opencv4`",  // OpenCV
``` -->

# Class
```cc
class Layer {  // Abstract base class for defining different types of layers in a neural network.
public:
    virtual std::vector<float> forward(const std::vector<float>& input) = 0;
    // `= 0`: This part makes the forward function a pure virtual function, which means that Layer is an abstract class and cannot be instantiated directly. Any class that inherits from Layer must provide its own implementation of the forward function.
    virtual ~Layer() = default;
    // `~Layer()`: This is the destructor for the Layer class. The destructor is invoked when an object of a class is destroyed, either explicitly using delete or when an object goes out of scope.
    // `= default`: This syntax indicates that the destructor is implicitly defined by the compiler. In this case, the default destructor does not perform any additional cleanup other than what is automatically provided by the compiler. This is fine here because the Layer class doesn't have any resources that need special cleanup (like dynamically allocated memory).
};
```

## Access Specifier
```cc
class DenseLayer : public Layer {
// `public`: 부모 클래스의 public 멤버와 protected 멤버가 자식 클래스에서 public으로 접근 가능합니다. 즉, 부모 클래스에서 public으로 정의된 함수나 변수를 자식 클래스가 그대로 public으로 사용할 수 있습니다.
// `private`: 부모 클래스의 public 멤버와 protected 멤버가 자식 클래스에서 private으로 접근 가능합니다. 자식 클래스 내부에서만 사용할 수 있고, 외부에서 접근할 수 없습니다.
// `protected`: 부모 클래스의 public 멤버와 protected 멤버가 자식 클래스에서 protected로 접근 가능합니다. 자식 클래스의 외부에서는 접근할 수 없고, 자식 클래스 내부에서만 접근 가능합니다.
```