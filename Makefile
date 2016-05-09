TARGET = CCL
OBJECT = main.o ccl_cpu.o ccl_gpu.o util.o

$(TARGET): $(OBJECT)
	nvcc -arch=sm_10 $(OBJECT) -o $(TARGET)
$(OBJECT): %.o: %.cu ccl_cpu.cuh ccl_gpu.cuh util.cuh
	nvcc -arch=sm_10 -c $< -o $@

.PHONY: clean
clean:
	-rm $(TARGET) $(OBJECT)