CXXFLAGS = -std=c++11 -Wall -O2 -Wno-parentheses -Wno-misleading-indentation -Wno-int-in-bool-context
CPPFLAGS = -I../include
LDFLAGS =
LIBS =

HEADERS = $(wildcard ../include/Spectra/MatOp/*.h) \
	$(wildcard ../include/Spectra/LinAlg/*.h) \
	$(wildcard ../include/Spectra/Util/*.h) \
	$(wildcard ../include/Spectra/*.h)


.PHONY: all test clean

all: QR.out Eigen.out SymEigs.out SymEigsShift.out GenEigs.out GenEigsRealShift.out GenEigsComplexShift.out \
	SymGEigsCholesky.out SymGEigsRegInv.out

test:
	-./QR.out
	-./Eigen.out
	-./SymEigs.out
	-./SymEigsShift.out
	-./GenEigs.out
	-./GenEigsRealShift.out
	-./GenEigsComplexShift.out
	-./SymGEigsCholesky.out
	-./SymGEigsRegInv.out

%.out: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	-rm *.out
