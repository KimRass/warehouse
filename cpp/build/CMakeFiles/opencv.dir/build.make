# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/jongbeomkim/Desktop/workspace/warehouse/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/jongbeomkim/Desktop/workspace/warehouse/cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/opencv.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/opencv.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/opencv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/opencv.dir/flags.make

CMakeFiles/opencv.dir/codegen:
.PHONY : CMakeFiles/opencv.dir/codegen

CMakeFiles/opencv.dir/opencv.cc.o: CMakeFiles/opencv.dir/flags.make
CMakeFiles/opencv.dir/opencv.cc.o: /Users/jongbeomkim/Desktop/workspace/warehouse/cpp/opencv.cc
CMakeFiles/opencv.dir/opencv.cc.o: CMakeFiles/opencv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/jongbeomkim/Desktop/workspace/warehouse/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/opencv.dir/opencv.cc.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv.dir/opencv.cc.o -MF CMakeFiles/opencv.dir/opencv.cc.o.d -o CMakeFiles/opencv.dir/opencv.cc.o -c /Users/jongbeomkim/Desktop/workspace/warehouse/cpp/opencv.cc

CMakeFiles/opencv.dir/opencv.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/opencv.dir/opencv.cc.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jongbeomkim/Desktop/workspace/warehouse/cpp/opencv.cc > CMakeFiles/opencv.dir/opencv.cc.i

CMakeFiles/opencv.dir/opencv.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/opencv.dir/opencv.cc.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jongbeomkim/Desktop/workspace/warehouse/cpp/opencv.cc -o CMakeFiles/opencv.dir/opencv.cc.s

# Object files for target opencv
opencv_OBJECTS = \
"CMakeFiles/opencv.dir/opencv.cc.o"

# External object files for target opencv
opencv_EXTERNAL_OBJECTS =

opencv: CMakeFiles/opencv.dir/opencv.cc.o
opencv: CMakeFiles/opencv.dir/build.make
opencv: /opt/homebrew/Cellar/opencv/4.10.0_12/lib/libopencv_core.dylib
opencv: /opt/homebrew/Cellar/opencv/4.10.0_12/lib/libopencv_imgcodecs.dylib
opencv: /opt/homebrew/Cellar/opencv/4.10.0_12/lib/libopencv_highgui.dylib
opencv: CMakeFiles/opencv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/jongbeomkim/Desktop/workspace/warehouse/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable opencv"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/opencv.dir/build: opencv
.PHONY : CMakeFiles/opencv.dir/build

CMakeFiles/opencv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/opencv.dir/cmake_clean.cmake
.PHONY : CMakeFiles/opencv.dir/clean

CMakeFiles/opencv.dir/depend:
	cd /Users/jongbeomkim/Desktop/workspace/warehouse/cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jongbeomkim/Desktop/workspace/warehouse/cpp /Users/jongbeomkim/Desktop/workspace/warehouse/cpp /Users/jongbeomkim/Desktop/workspace/warehouse/cpp/build /Users/jongbeomkim/Desktop/workspace/warehouse/cpp/build /Users/jongbeomkim/Desktop/workspace/warehouse/cpp/build/CMakeFiles/opencv.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/opencv.dir/depend

