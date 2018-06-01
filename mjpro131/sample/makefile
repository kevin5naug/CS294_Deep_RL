all:
	clang -O2 -std=c++11 -I../include simulate.cpp -lstdc++ ../bin/libmujoco131.dylib ../bin/libglfw.3.dylib -o ../bin/simulate
	clang -O2 -std=c++11 -I../include test.cpp -lstdc++ ../bin/libmujoco131.dylib -o ../bin/test
	clang -O2 -std=c++11 -I../include compile.cpp -lstdc++ ../bin/libmujoco131.dylib -o ../bin/compile
