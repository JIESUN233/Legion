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
CMAKE_SOURCE_DIR = /home/szc/pcm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szc/pcm/build

# Include any dependencies generated for this target.
include src/CMakeFiles/pcm-pcie.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/pcm-pcie.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/pcm-pcie.dir/flags.make

src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o: src/CMakeFiles/pcm-pcie.dir/flags.make
src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o: ../src/pcm-pcie.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/pcm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o"
	cd /home/szc/pcm/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o -c /home/szc/pcm/src/pcm-pcie.cpp

src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.i"
	cd /home/szc/pcm/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/szc/pcm/src/pcm-pcie.cpp > CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.i

src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.s"
	cd /home/szc/pcm/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/szc/pcm/src/pcm-pcie.cpp -o CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.s

src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o.requires:

.PHONY : src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o.requires

src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o.provides: src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/pcm-pcie.dir/build.make src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o.provides.build
.PHONY : src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o.provides

src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o.provides.build: src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o


# Object files for target pcm-pcie
pcm__pcie_OBJECTS = \
"CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o"

# External object files for target pcm-pcie
pcm__pcie_EXTERNAL_OBJECTS =

bin/pcm-pcie: src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o
bin/pcm-pcie: src/CMakeFiles/pcm-pcie.dir/build.make
bin/pcm-pcie: src/libpcm.a
bin/pcm-pcie: src/CMakeFiles/pcm-pcie.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/szc/pcm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/pcm-pcie"
	cd /home/szc/pcm/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pcm-pcie.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/pcm-pcie.dir/build: bin/pcm-pcie

.PHONY : src/CMakeFiles/pcm-pcie.dir/build

src/CMakeFiles/pcm-pcie.dir/requires: src/CMakeFiles/pcm-pcie.dir/pcm-pcie.cpp.o.requires

.PHONY : src/CMakeFiles/pcm-pcie.dir/requires

src/CMakeFiles/pcm-pcie.dir/clean:
	cd /home/szc/pcm/build/src && $(CMAKE_COMMAND) -P CMakeFiles/pcm-pcie.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/pcm-pcie.dir/clean

src/CMakeFiles/pcm-pcie.dir/depend:
	cd /home/szc/pcm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/pcm /home/szc/pcm/src /home/szc/pcm/build /home/szc/pcm/build/src /home/szc/pcm/build/src/CMakeFiles/pcm-pcie.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/pcm-pcie.dir/depend

