CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17
TARGET = out.exe

all: clean $(TARGET)

$(TARGET): out.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) out.cpp

clean:
	rm -f *.o $(TARGET)
