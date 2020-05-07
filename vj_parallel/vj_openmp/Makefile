GCC ?= g++

HEADERS := image.h haar.h stdio-wrapper.h

all: build

build: vj

image.o: image.c $(HEADERS)
	$(GCC) -fopenmp -o $@ -c $<

stdio.o: stdio-wrapper.c $(HEADERS)
	$(GCC) -fopenmp -o $@ -c $<

main.o: main.cpp $(HEADERS)
	$(GCC) -fopenmp -std=c++11 -o $@ -c $<

haar.o: haar.cpp $(HEADERS)
	$(GCC) -fopenmp -std=c++11 -o $@ -c $< 

rectangles.o: rectangles.cpp $(HEADERS)
	$(GCC) -fopenmp -o $@ -c $<

vj: main.o haar.o image.o stdio-wrapper.o rectangles.o
	$(GCC) -fopenmp -o $@ $+ $(LDFLAGS)

run: build
	./vj

clean:
	rm -f vj *.o Output.pgm
