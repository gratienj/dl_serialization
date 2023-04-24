all: main

CXXFLAG :=
INCLUDE := -I$(TF_INCLUDE)
LIB := -l$(TF_LFLAG)
LFLAG := -L$(TF_LIB)

main:
	$(CXX) $(CXXFLAG) -O3 $(INCLUDE) $(LFLAG) -c TFcaller.cpp
	$(CXX) $(CXXFLAG) -O3 $(INCLUDE) $(LFLAG) -o mainProgram main.cpp TFcaller.o $(LIB)

# debug:
# 	$(CXX) $(CXXFLAG) -g $(INCLUDE) $(LFLAG) -c TFcaller.cpp
# 	$(CXX) $(CXXFLAG) -g $(INCLUDE) $(LFLAG) -o mainProgram main.cpp TFcaller.o $(LIB)

clean:
	rm -f mainProgram *.o