# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jianyuzh/ws/llama.cpp/develop/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jianyuzh/ws/llama.cpp/develop/examples/sycl

# Include any dependencies generated for this target.
include quantize/CMakeFiles/quantize.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include quantize/CMakeFiles/quantize.dir/compiler_depend.make

# Include the progress variables for this target.
include quantize/CMakeFiles/quantize.dir/progress.make

# Include the compile flags for this target's objects.
include quantize/CMakeFiles/quantize.dir/flags.make

quantize/CMakeFiles/quantize.dir/quantize.o: quantize/CMakeFiles/quantize.dir/flags.make
quantize/CMakeFiles/quantize.dir/quantize.o: ../quantize/quantize.cpp
quantize/CMakeFiles/quantize.dir/quantize.o: quantize/CMakeFiles/quantize.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object quantize/CMakeFiles/quantize.dir/quantize.o"
	cd /home/jianyuzh/ws/llama.cpp/develop/examples/sycl/quantize && /opt/intel/oneapi/compiler/2024.0/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT quantize/CMakeFiles/quantize.dir/quantize.o -MF CMakeFiles/quantize.dir/quantize.o.d -o CMakeFiles/quantize.dir/quantize.o -c /home/jianyuzh/ws/llama.cpp/develop/examples/quantize/quantize.cpp

quantize/CMakeFiles/quantize.dir/quantize.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quantize.dir/quantize.i"
	cd /home/jianyuzh/ws/llama.cpp/develop/examples/sycl/quantize && /opt/intel/oneapi/compiler/2024.0/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jianyuzh/ws/llama.cpp/develop/examples/quantize/quantize.cpp > CMakeFiles/quantize.dir/quantize.i

quantize/CMakeFiles/quantize.dir/quantize.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quantize.dir/quantize.s"
	cd /home/jianyuzh/ws/llama.cpp/develop/examples/sycl/quantize && /opt/intel/oneapi/compiler/2024.0/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jianyuzh/ws/llama.cpp/develop/examples/quantize/quantize.cpp -o CMakeFiles/quantize.dir/quantize.s

# Object files for target quantize
quantize_OBJECTS = \
"CMakeFiles/quantize.dir/quantize.o"

# External object files for target quantize
quantize_EXTERNAL_OBJECTS =

quantize/quantize: quantize/CMakeFiles/quantize.dir/quantize.o
quantize/quantize: quantize/CMakeFiles/quantize.dir/build.make
quantize/quantize: quantize/CMakeFiles/quantize.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jianyuzh/ws/llama.cpp/develop/examples/sycl/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable quantize"
	cd /home/jianyuzh/ws/llama.cpp/develop/examples/sycl/quantize && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/quantize.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
quantize/CMakeFiles/quantize.dir/build: quantize/quantize
.PHONY : quantize/CMakeFiles/quantize.dir/build

quantize/CMakeFiles/quantize.dir/clean:
	cd /home/jianyuzh/ws/llama.cpp/develop/examples/sycl/quantize && $(CMAKE_COMMAND) -P CMakeFiles/quantize.dir/cmake_clean.cmake
.PHONY : quantize/CMakeFiles/quantize.dir/clean

quantize/CMakeFiles/quantize.dir/depend:
	cd /home/jianyuzh/ws/llama.cpp/develop/examples/sycl && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jianyuzh/ws/llama.cpp/develop/examples /home/jianyuzh/ws/llama.cpp/develop/examples/quantize /home/jianyuzh/ws/llama.cpp/develop/examples/sycl /home/jianyuzh/ws/llama.cpp/develop/examples/sycl/quantize /home/jianyuzh/ws/llama.cpp/develop/examples/sycl/quantize/CMakeFiles/quantize.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : quantize/CMakeFiles/quantize.dir/depend

