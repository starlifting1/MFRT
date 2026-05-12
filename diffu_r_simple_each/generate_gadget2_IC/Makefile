CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17
TARGET = generate_IC.exe

all: clean $(TARGET)

$(TARGET): generate_IC.cpp Gadget2FormatData_io.cpp generate_gadget2_IC.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) generate_IC.cpp ../diffu.cpp ../generate_samples.cpp Gadget2FormatData_io.cpp generate_gadget2_IC.cpp

clean:
	rm -f *.o $(TARGET)
