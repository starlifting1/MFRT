CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17
TARGET = out.exe
TARGET2 = out_vel.exe

all: clean $(TARGET) $(TARGET2)

$(TARGET): main.cpp diffu.cpp generate_samples.cpp direct_summation.cpp Tree_method.cpp tree_vel.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp diffu.cpp generate_samples.cpp direct_summation.cpp Tree_method.cpp tree_vel.cpp

$(TARGET2): main_vel.cpp diffu.cpp generate_samples.cpp direct_summation.cpp Tree_method.cpp tree_vel.cpp fractal_dim.cpp whitening_recoloring.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET2) main_vel.cpp diffu.cpp generate_samples.cpp direct_summation.cpp Tree_method.cpp tree_vel.cpp fractal_dim.cpp whitening_recoloring.cpp

clean:
	rm -f *.o $(TARGET) $(TARGET2)
