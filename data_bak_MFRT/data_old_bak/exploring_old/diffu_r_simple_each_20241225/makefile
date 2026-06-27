CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17
TARGET = out.exe

all: clean $(TARGET)

$(TARGET): main.cpp diffu.cpp Tree_method.cpp generate_samples.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp diffu.cpp Tree_method.cpp generate_samples.cpp

clean:
	rm -f *.o $(TARGET)
