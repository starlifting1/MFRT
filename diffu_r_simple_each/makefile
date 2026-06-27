CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17
TARGET = out.exe
TARGET2 = out_vel.exe
TARGET_POS_FRACTAL = main_pos_fractal.exe

.PHONY: all clean main_pos_fractal

all: clean $(TARGET) $(TARGET2) $(TARGET_POS_FRACTAL)

$(TARGET): main.cpp diffu.cpp generate_samples.cpp direct_summation.cpp Tree_method.cpp tree_vel.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp diffu.cpp generate_samples.cpp direct_summation.cpp Tree_method.cpp tree_vel.cpp

$(TARGET2): main_vel.cpp diffu.cpp generate_samples.cpp direct_summation.cpp Tree_method.cpp tree_vel.cpp fractal_dim.cpp whitening_recoloring.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET2) main_vel.cpp diffu.cpp generate_samples.cpp direct_summation.cpp Tree_method.cpp tree_vel.cpp fractal_dim.cpp whitening_recoloring.cpp

main_pos_fractal: $(TARGET_POS_FRACTAL)

$(TARGET_POS_FRACTAL): main_pos_fractal.cpp diffu.cpp generate_samples.cpp direct_summation.cpp Tree_method.cpp tree_vel.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_POS_FRACTAL) main_pos_fractal.cpp diffu.cpp generate_samples.cpp direct_summation.cpp Tree_method.cpp tree_vel.cpp

clean:
	rm -f *.o $(TARGET) $(TARGET2) $(TARGET_POS_FRACTAL)
