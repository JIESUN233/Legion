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
include src/CMakeFiles/pcm-lspci.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/pcm-lspci.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/pcm-lspci.dir/flags.make

src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o: src/CMakeFiles/pcm-lspci.dir/flags.make
src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o: ../src/pcm-lspci.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/pcm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o"
	cd /home/szc/pcm/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o -c /home/szc/pcm/src/pcm-lspci.cpp

src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.i"
	cd /home/szc/pcm/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/szc/pcm/src/pcm-lspci.cpp > CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.i

src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.s"
	cd /home/szc/pcm/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/szc/pcm/src/pcm-lspci.cpp -o CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.s

src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o.requires:

.PHONY : src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o.requires

src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o.provides: src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/pcm-lspci.dir/build.make src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o.provides.build
.PHONY : src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o.provides

src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o.provides.build: src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o


# Object files for target pcm-lspci
pcm__lspci_OBJECTS = \
"CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o"

# External object files for target pcm-lspci
pcm__lspci_EXTERNAL_OBJECTS =

bin/pcm-lspci: src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o
bin/pcm-lspci: src/CMakeFiles/pcm-lspci.dir/build.make
bin/pcm-lspci: src/libpcm.a
bin/pcm-lspci: src/CMakeFiles/pcm-lspci.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/szc/pcm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/pcm-lspci"
	cd /home/szc/pcm/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pcm-lspci.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/pcm-lspci.dir/build: bin/pcm-lspci

.PHONY : src/CMakeFiles/pcm-lspci.dir/build

src/CMakeFiles/pcm-lspci.dir/requires: src/CMakeFiles/pcm-lspci.dir/pcm-lspci.cpp.o.requires

.PHONY : src/CMakeFiles/pcm-lspci.dir/requires

src/CMakeFiles/pcm-lspci.dir/clean:
	cd /home/szc/pcm/build/src && $(CMAKE_COMMAND) -P CMakeFiles/pcm-lspci.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/pcm-lspci.dir/clean

src/CMakeFiles/pcm-lspci.dir/depend:
	cd /home/szc/pcm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/pcm /home/szc/pcm/src /home/szc/pcm/build /home/szc/pcm/build/src /home/szc/pcm/build/src/CMakeFiles/pcm-lspci.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/pcm-lspci.dir/depend

