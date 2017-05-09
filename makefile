crle: *.cu */*.h
	nvcc -arch=sm_20 -o crl kernel.cu 
