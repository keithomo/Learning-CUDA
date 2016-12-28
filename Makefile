
COMPUTECAP=52
LDFLGS=$(CUDA)/lib64

rcog_rsk: rcog_rsk.cu
	nvcc -arch=compute_$(COMPUTECAP) -g -o rcog_rsk  rcog_rsk.cu -L$(LDFLGS) -lcufft

clean:
	rm -f rcog_rsk

info:
	@echo "CUDA=" $(CUDA)
	@echo "LD_LIBRARY_PATH=" $(LD_LIBRARY_PATH)
