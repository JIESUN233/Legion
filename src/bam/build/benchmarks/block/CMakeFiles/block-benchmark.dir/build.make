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

# Utility rule file for block-benchmark.

# Include the progress variables for this target.
include benchmarks/block/CMakeFiles/block-benchmark.dir/progress.make

benchmarks/block/CMakeFiles/block-benchmark: bin/nvm-block-bench


block-benchmark: benchmarks/block/CMakeFiles/block-benchmark
block-benchmark: benchmarks/block/CMakeFiles/block-benchmark.dir/build.make

.PHONY : block-benchmark

# Rule to build all files generated by this target.
benchmarks/block/CMakeFiles/block-benchmark.dir/build: block-benchmark

.PHONY : benchmarks/block/CMakeFiles/block-benchmark.dir/build

benchmarks/block/CMakeFiles/block-benchmark.dir/clean:
	cd /home/sunjie/bam/build/benchmarks/block && $(CMAKE_COMMAND) -P CMakeFiles/block-benchmark.dir/cmake_clean.cmake
.PHONY : benchmarks/block/CMakeFiles/block-benchmark.dir/clean

benchmarks/block/CMakeFiles/block-benchmark.dir/depend:
	cd /home/sunjie/bam/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sunjie/bam /home/sunjie/bam/benchmarks/block /home/sunjie/bam/build /home/sunjie/bam/build/benchmarks/block /home/sunjie/bam/build/benchmarks/block/CMakeFiles/block-benchmark.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : benchmarks/block/CMakeFiles/block-benchmark.dir/depend

