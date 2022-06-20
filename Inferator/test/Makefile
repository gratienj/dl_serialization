all: main

CXXFLAG :=
INCLUDE := -I$(HDF5_INC) -I$(EIGEN_INC) -I$(JSON_INC)
LIB := -lhdf5
LFLAG := -L$(HDF5_LIB)

main:
	$(CXX) $(CXXFLAG) -O3 $(INCLUDE) $(LFLAG) -c ../src/inference.cpp
	$(CXX) $(CXXFLAG) -O3 $(INCLUDE) $(LFLAG) -o mainProgram main.cpp inference.o $(LIB)

debug:
	$(CXX) $(CXXFLAG) -g $(INCLUDE) $(LFLAG) -c ../src/inference.cpp
	$(CXX) $(CXXFLAG) -g $(INCLUDE) $(LFLAG) -o mainProgram main.cpp inference.o $(LIB)

clean:
	rm -f mainProgram *.o