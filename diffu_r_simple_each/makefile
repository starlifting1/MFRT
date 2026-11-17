CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17
CPPFLAGS ?= -I/usr/include/eigen3
.PHONY: all clean

all: out_revision.exe

out_revision.exe: run_revision.cpp diffu.cpp generate_samples.cpp direct_summation.cpp Tree_method.cpp tree_vel.cpp fractal_dim.cpp whitening_recoloring.cpp $(wildcard *.h *.hpp) makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ run_revision.cpp diffu.cpp generate_samples.cpp direct_summation.cpp Tree_method.cpp tree_vel.cpp fractal_dim.cpp whitening_recoloring.cpp

clean:
	rm -f *.o out_revision.exe
