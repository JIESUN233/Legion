# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sunjie/bam

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sunjie/bam/build

# Include any dependencies generated for this target.
include CMakeFiles/libnvm.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/libnvm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/libnvm.dir/flags.make

CMakeFiles/libnvm.dir/src/admin.cpp.o: CMakeFiles/libnvm.dir/flags.make
CMakeFiles/libnvm.dir/src/admin.cpp.o: ../src/admin.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sunjie/bam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/libnvm.dir/src/admin.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libnvm.dir/src/admin.cpp.o -c /home/sunjie/bam/src/admin.cpp

CMakeFiles/libnvm.dir/src/admin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/admin.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunjie/bam/src/admin.cpp > CMakeFiles/libnvm.dir/src/admin.cpp.i

CMakeFiles/libnvm.dir/src/admin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/admin.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunjie/bam/src/admin.cpp -o CMakeFiles/libnvm.dir/src/admin.cpp.s

CMakeFiles/libnvm.dir/src/admin.cpp.o.requires:

.PHONY : CMakeFiles/libnvm.dir/src/admin.cpp.o.requires

CMakeFiles/libnvm.dir/src/admin.cpp.o.provides: CMakeFiles/libnvm.dir/src/admin.cpp.o.requires
	$(MAKE) -f CMakeFiles/libnvm.dir/build.make CMakeFiles/libnvm.dir/src/admin.cpp.o.provides.build
.PHONY : CMakeFiles/libnvm.dir/src/admin.cpp.o.provides

CMakeFiles/libnvm.dir/src/admin.cpp.o.provides.build: CMakeFiles/libnvm.dir/src/admin.cpp.o


CMakeFiles/libnvm.dir/src/ctrl.cpp.o: CMakeFiles/libnvm.dir/flags.make
CMakeFiles/libnvm.dir/src/ctrl.cpp.o: ../src/ctrl.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sunjie/bam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/libnvm.dir/src/ctrl.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libnvm.dir/src/ctrl.cpp.o -c /home/sunjie/bam/src/ctrl.cpp

CMakeFiles/libnvm.dir/src/ctrl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/ctrl.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunjie/bam/src/ctrl.cpp > CMakeFiles/libnvm.dir/src/ctrl.cpp.i

CMakeFiles/libnvm.dir/src/ctrl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/ctrl.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunjie/bam/src/ctrl.cpp -o CMakeFiles/libnvm.dir/src/ctrl.cpp.s

CMakeFiles/libnvm.dir/src/ctrl.cpp.o.requires:

.PHONY : CMakeFiles/libnvm.dir/src/ctrl.cpp.o.requires

CMakeFiles/libnvm.dir/src/ctrl.cpp.o.provides: CMakeFiles/libnvm.dir/src/ctrl.cpp.o.requires
	$(MAKE) -f CMakeFiles/libnvm.dir/build.make CMakeFiles/libnvm.dir/src/ctrl.cpp.o.provides.build
.PHONY : CMakeFiles/libnvm.dir/src/ctrl.cpp.o.provides

CMakeFiles/libnvm.dir/src/ctrl.cpp.o.provides.build: CMakeFiles/libnvm.dir/src/ctrl.cpp.o


CMakeFiles/libnvm.dir/src/dma.cpp.o: CMakeFiles/libnvm.dir/flags.make
CMakeFiles/libnvm.dir/src/dma.cpp.o: ../src/dma.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sunjie/bam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/libnvm.dir/src/dma.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libnvm.dir/src/dma.cpp.o -c /home/sunjie/bam/src/dma.cpp

CMakeFiles/libnvm.dir/src/dma.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/dma.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunjie/bam/src/dma.cpp > CMakeFiles/libnvm.dir/src/dma.cpp.i

CMakeFiles/libnvm.dir/src/dma.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/dma.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunjie/bam/src/dma.cpp -o CMakeFiles/libnvm.dir/src/dma.cpp.s

CMakeFiles/libnvm.dir/src/dma.cpp.o.requires:

.PHONY : CMakeFiles/libnvm.dir/src/dma.cpp.o.requires

CMakeFiles/libnvm.dir/src/dma.cpp.o.provides: CMakeFiles/libnvm.dir/src/dma.cpp.o.requires
	$(MAKE) -f CMakeFiles/libnvm.dir/build.make CMakeFiles/libnvm.dir/src/dma.cpp.o.provides.build
.PHONY : CMakeFiles/libnvm.dir/src/dma.cpp.o.provides

CMakeFiles/libnvm.dir/src/dma.cpp.o.provides.build: CMakeFiles/libnvm.dir/src/dma.cpp.o


CMakeFiles/libnvm.dir/src/error.cpp.o: CMakeFiles/libnvm.dir/flags.make
CMakeFiles/libnvm.dir/src/error.cpp.o: ../src/error.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sunjie/bam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/libnvm.dir/src/error.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libnvm.dir/src/error.cpp.o -c /home/sunjie/bam/src/error.cpp

CMakeFiles/libnvm.dir/src/error.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/error.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunjie/bam/src/error.cpp > CMakeFiles/libnvm.dir/src/error.cpp.i

CMakeFiles/libnvm.dir/src/error.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/error.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunjie/bam/src/error.cpp -o CMakeFiles/libnvm.dir/src/error.cpp.s

CMakeFiles/libnvm.dir/src/error.cpp.o.requires:

.PHONY : CMakeFiles/libnvm.dir/src/error.cpp.o.requires

CMakeFiles/libnvm.dir/src/error.cpp.o.provides: CMakeFiles/libnvm.dir/src/error.cpp.o.requires
	$(MAKE) -f CMakeFiles/libnvm.dir/build.make CMakeFiles/libnvm.dir/src/error.cpp.o.provides.build
.PHONY : CMakeFiles/libnvm.dir/src/error.cpp.o.provides

CMakeFiles/libnvm.dir/src/error.cpp.o.provides.build: CMakeFiles/libnvm.dir/src/error.cpp.o


CMakeFiles/libnvm.dir/src/mutex.cpp.o: CMakeFiles/libnvm.dir/flags.make
CMakeFiles/libnvm.dir/src/mutex.cpp.o: ../src/mutex.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sunjie/bam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/libnvm.dir/src/mutex.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libnvm.dir/src/mutex.cpp.o -c /home/sunjie/bam/src/mutex.cpp

CMakeFiles/libnvm.dir/src/mutex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/mutex.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunjie/bam/src/mutex.cpp > CMakeFiles/libnvm.dir/src/mutex.cpp.i

CMakeFiles/libnvm.dir/src/mutex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/mutex.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunjie/bam/src/mutex.cpp -o CMakeFiles/libnvm.dir/src/mutex.cpp.s

CMakeFiles/libnvm.dir/src/mutex.cpp.o.requires:

.PHONY : CMakeFiles/libnvm.dir/src/mutex.cpp.o.requires

CMakeFiles/libnvm.dir/src/mutex.cpp.o.provides: CMakeFiles/libnvm.dir/src/mutex.cpp.o.requires
	$(MAKE) -f CMakeFiles/libnvm.dir/build.make CMakeFiles/libnvm.dir/src/mutex.cpp.o.provides.build
.PHONY : CMakeFiles/libnvm.dir/src/mutex.cpp.o.provides

CMakeFiles/libnvm.dir/src/mutex.cpp.o.provides.build: CMakeFiles/libnvm.dir/src/mutex.cpp.o


CMakeFiles/libnvm.dir/src/queue.cpp.o: CMakeFiles/libnvm.dir/flags.make
CMakeFiles/libnvm.dir/src/queue.cpp.o: ../src/queue.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sunjie/bam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/libnvm.dir/src/queue.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libnvm.dir/src/queue.cpp.o -c /home/sunjie/bam/src/queue.cpp

CMakeFiles/libnvm.dir/src/queue.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/queue.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunjie/bam/src/queue.cpp > CMakeFiles/libnvm.dir/src/queue.cpp.i

CMakeFiles/libnvm.dir/src/queue.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/queue.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunjie/bam/src/queue.cpp -o CMakeFiles/libnvm.dir/src/queue.cpp.s

CMakeFiles/libnvm.dir/src/queue.cpp.o.requires:

.PHONY : CMakeFiles/libnvm.dir/src/queue.cpp.o.requires

CMakeFiles/libnvm.dir/src/queue.cpp.o.provides: CMakeFiles/libnvm.dir/src/queue.cpp.o.requires
	$(MAKE) -f CMakeFiles/libnvm.dir/build.make CMakeFiles/libnvm.dir/src/queue.cpp.o.provides.build
.PHONY : CMakeFiles/libnvm.dir/src/queue.cpp.o.provides

CMakeFiles/libnvm.dir/src/queue.cpp.o.provides.build: CMakeFiles/libnvm.dir/src/queue.cpp.o


CMakeFiles/libnvm.dir/src/rpc.cpp.o: CMakeFiles/libnvm.dir/flags.make
CMakeFiles/libnvm.dir/src/rpc.cpp.o: ../src/rpc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sunjie/bam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/libnvm.dir/src/rpc.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libnvm.dir/src/rpc.cpp.o -c /home/sunjie/bam/src/rpc.cpp

CMakeFiles/libnvm.dir/src/rpc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/rpc.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunjie/bam/src/rpc.cpp > CMakeFiles/libnvm.dir/src/rpc.cpp.i

CMakeFiles/libnvm.dir/src/rpc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/rpc.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunjie/bam/src/rpc.cpp -o CMakeFiles/libnvm.dir/src/rpc.cpp.s

CMakeFiles/libnvm.dir/src/rpc.cpp.o.requires:

.PHONY : CMakeFiles/libnvm.dir/src/rpc.cpp.o.requires

CMakeFiles/libnvm.dir/src/rpc.cpp.o.provides: CMakeFiles/libnvm.dir/src/rpc.cpp.o.requires
	$(MAKE) -f CMakeFiles/libnvm.dir/build.make CMakeFiles/libnvm.dir/src/rpc.cpp.o.provides.build
.PHONY : CMakeFiles/libnvm.dir/src/rpc.cpp.o.provides

CMakeFiles/libnvm.dir/src/rpc.cpp.o.provides.build: CMakeFiles/libnvm.dir/src/rpc.cpp.o


CMakeFiles/libnvm.dir/src/linux/device.cpp.o: CMakeFiles/libnvm.dir/flags.make
CMakeFiles/libnvm.dir/src/linux/device.cpp.o: ../src/linux/device.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sunjie/bam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/libnvm.dir/src/linux/device.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libnvm.dir/src/linux/device.cpp.o -c /home/sunjie/bam/src/linux/device.cpp

CMakeFiles/libnvm.dir/src/linux/device.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/linux/device.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunjie/bam/src/linux/device.cpp > CMakeFiles/libnvm.dir/src/linux/device.cpp.i

CMakeFiles/libnvm.dir/src/linux/device.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/linux/device.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunjie/bam/src/linux/device.cpp -o CMakeFiles/libnvm.dir/src/linux/device.cpp.s

CMakeFiles/libnvm.dir/src/linux/device.cpp.o.requires:

.PHONY : CMakeFiles/libnvm.dir/src/linux/device.cpp.o.requires

CMakeFiles/libnvm.dir/src/linux/device.cpp.o.provides: CMakeFiles/libnvm.dir/src/linux/device.cpp.o.requires
	$(MAKE) -f CMakeFiles/libnvm.dir/build.make CMakeFiles/libnvm.dir/src/linux/device.cpp.o.provides.build
.PHONY : CMakeFiles/libnvm.dir/src/linux/device.cpp.o.provides

CMakeFiles/libnvm.dir/src/linux/device.cpp.o.provides.build: CMakeFiles/libnvm.dir/src/linux/device.cpp.o


CMakeFiles/libnvm.dir/src/linux/dma.cpp.o: CMakeFiles/libnvm.dir/flags.make
CMakeFiles/libnvm.dir/src/linux/dma.cpp.o: ../src/linux/dma.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sunjie/bam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/libnvm.dir/src/linux/dma.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libnvm.dir/src/linux/dma.cpp.o -c /home/sunjie/bam/src/linux/dma.cpp

CMakeFiles/libnvm.dir/src/linux/dma.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/linux/dma.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sunjie/bam/src/linux/dma.cpp > CMakeFiles/libnvm.dir/src/linux/dma.cpp.i

CMakeFiles/libnvm.dir/src/linux/dma.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/linux/dma.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sunjie/bam/src/linux/dma.cpp -o CMakeFiles/libnvm.dir/src/linux/dma.cpp.s

CMakeFiles/libnvm.dir/src/linux/dma.cpp.o.requires:

.PHONY : CMakeFiles/libnvm.dir/src/linux/dma.cpp.o.requires

CMakeFiles/libnvm.dir/src/linux/dma.cpp.o.provides: CMakeFiles/libnvm.dir/src/linux/dma.cpp.o.requires
	$(MAKE) -f CMakeFiles/libnvm.dir/build.make CMakeFiles/libnvm.dir/src/linux/dma.cpp.o.provides.build
.PHONY : CMakeFiles/libnvm.dir/src/linux/dma.cpp.o.provides

CMakeFiles/libnvm.dir/src/linux/dma.cpp.o.provides.build: CMakeFiles/libnvm.dir/src/linux/dma.cpp.o


# Object files for target libnvm
libnvm_OBJECTS = \
"CMakeFiles/libnvm.dir/src/admin.cpp.o" \
"CMakeFiles/libnvm.dir/src/ctrl.cpp.o" \
"CMakeFiles/libnvm.dir/src/dma.cpp.o" \
"CMakeFiles/libnvm.dir/src/error.cpp.o" \
"CMakeFiles/libnvm.dir/src/mutex.cpp.o" \
"CMakeFiles/libnvm.dir/src/queue.cpp.o" \
"CMakeFiles/libnvm.dir/src/rpc.cpp.o" \
"CMakeFiles/libnvm.dir/src/linux/device.cpp.o" \
"CMakeFiles/libnvm.dir/src/linux/dma.cpp.o"

# External object files for target libnvm
libnvm_EXTERNAL_OBJECTS =

lib/libnvm.so: CMakeFiles/libnvm.dir/src/admin.cpp.o
lib/libnvm.so: CMakeFiles/libnvm.dir/src/ctrl.cpp.o
lib/libnvm.so: CMakeFiles/libnvm.dir/src/dma.cpp.o
lib/libnvm.so: CMakeFiles/libnvm.dir/src/error.cpp.o
lib/libnvm.so: CMakeFiles/libnvm.dir/src/mutex.cpp.o
lib/libnvm.so: CMakeFiles/libnvm.dir/src/queue.cpp.o
lib/libnvm.so: CMakeFiles/libnvm.dir/src/rpc.cpp.o
lib/libnvm.so: CMakeFiles/libnvm.dir/src/linux/device.cpp.o
lib/libnvm.so: CMakeFiles/libnvm.dir/src/linux/dma.cpp.o
lib/libnvm.so: CMakeFiles/libnvm.dir/build.make
lib/libnvm.so: /usr/local/cuda-11.7/lib64/libcudart_static.a
lib/libnvm.so: /usr/lib/x86_64-linux-gnu/librt.so
lib/libnvm.so: /usr/local/cuda-11.7/lib64/libcudart_static.a
lib/libnvm.so: /usr/lib/x86_64-linux-gnu/librt.so
lib/libnvm.so: CMakeFiles/libnvm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sunjie/bam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX shared library lib/libnvm.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libnvm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/libnvm.dir/build: lib/libnvm.so

.PHONY : CMakeFiles/libnvm.dir/build

CMakeFiles/libnvm.dir/requires: CMakeFiles/libnvm.dir/src/admin.cpp.o.requires
CMakeFiles/libnvm.dir/requires: CMakeFiles/libnvm.dir/src/ctrl.cpp.o.requires
CMakeFiles/libnvm.dir/requires: CMakeFiles/libnvm.dir/src/dma.cpp.o.requires
CMakeFiles/libnvm.dir/requires: CMakeFiles/libnvm.dir/src/error.cpp.o.requires
CMakeFiles/libnvm.dir/requires: CMakeFiles/libnvm.dir/src/mutex.cpp.o.requires
CMakeFiles/libnvm.dir/requires: CMakeFiles/libnvm.dir/src/queue.cpp.o.requires
CMakeFiles/libnvm.dir/requires: CMakeFiles/libnvm.dir/src/rpc.cpp.o.requires
CMakeFiles/libnvm.dir/requires: CMakeFiles/libnvm.dir/src/linux/device.cpp.o.requires
CMakeFiles/libnvm.dir/requires: CMakeFiles/libnvm.dir/src/linux/dma.cpp.o.requires

.PHONY : CMakeFiles/libnvm.dir/requires

CMakeFiles/libnvm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libnvm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libnvm.dir/clean

CMakeFiles/libnvm.dir/depend:
	cd /home/sunjie/bam/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sunjie/bam /home/sunjie/bam /home/sunjie/bam/build /home/sunjie/bam/build /home/sunjie/bam/build/CMakeFiles/libnvm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libnvm.dir/depend
