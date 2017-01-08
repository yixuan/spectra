CXXFLAGS = -std=c++11 -Wall -O2
CPPFLAGS = -I../include -I/usr/local/include/eigen3
LDFLAGS =
LIBS =

HEADERS = $(wildcard ../include/MatOp/*.h) $(wildcard ../include/LinAlg/*.h) $(wildcard ../include/Util/*.h) $(wildcard ../include/*.h)


.PHONY: all test clean

all: QR Eigen SymEigs SymEigsShift SymGEigsCholesky GenEigs GenEigsRealShift GenEigsComplexShift

test:
	-./QR
	-./Eigen
	-./SymEigs
	-./SymEigsShift
	-./SymGEigsCholesky
	-./GenEigs
	-./GenEigsRealShift
	-./GenEigsComplexShift

clean:
	-rm 
