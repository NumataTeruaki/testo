main: 
	g++ -std=c++14 -lm -I/usr/include/eigen3 -o program ./*.cpp `pkg-config opencv --cflags --libs`

ie:
	g++ -std=c++14 -lm -I./eigen3/ -o test ./*.cpp ./*.h

clear:
	rm ./convertEH
	rm ./convertEH.exe
	rm ./convertEH.o

retry:
	make clear | make
