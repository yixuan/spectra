CXXFLAGS = -std=c++11 -Wall -O2 -Wno-parentheses -Wno-misleading-indentation -Wno-int-in-bool-context
CPPFLAGS = -I../include
LDFLAGS =
LIBS =

HEADERS = $(wildcard ../include/Spectra/MatOp/*.h) \
	$(wildcard ../include/Spectra/MatOp/internal/*.h) \
	$(wildcard ../include/Spectra/LinAlg/*.h) \
	$(wildcard ../include/Spectra/Util/*.h) \
	$(wildcard ../include/Spectra/*.h) \
	$(wildcard ../include/Spectra/contrib/*.h)


.PHONY: all test clean

all: QR.out Eigen.out BKLDLT.out SymEigs.out SymEigsShift.out GenEigs.out GenEigsRealShift.out GenEigsComplexShift.out \
	SymGEigsCholesky.out SymGEigsRegInv.out SymGEigsShift.out SVD.out

test:
	-./QR.out
	-./Eigen.out
	-./BKLDLT.out
	-./SymEigs.out
	-./SymEigsShift.out
	-./GenEigs.out
	-./GenEigsRealShift.out
	-./GenEigsComplexShift.out
	-./SymGEigsCholesky.out
	-./SymGEigsRegInv.out
	-./SymGEigsShift.out
	-./SVD.out

tests-main.o: tests-main.cpp catch.hpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

%.out: %.cpp $(HEADERS) tests-main.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< tests-main.o -o $@ $(LDFLAGS) $(LIBS)

clean:
	-rm *.out
	-rm *.o

