NVCC = nvcc
CXX = nvcc
Compute_Capability = sm_80
CFLAGS = -std=c++14 -O3 
#--generate-line-info
NVFLAGS = -gencode=arch=compute_61,code=$(Compute_Capability) -Xptxas=-v -O3 -Xcompiler -Wall -D_FORCE_INLINES --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr
# A too high compute would enable Independent Thread Scheduling, which may lead to performance degradation for some kernels.
# compute_61 is safe, compute_70 is not.
INCLUDES = -Iinclude/ -Ikernels/
BINDIR = bin

TARGET = spgemm
UTILS = convert

CU_SRCS = $(wildcard src/*.cu)               # All .cu files in src/
CPP_SRCS = $(wildcard src/*.cpp)  

CU_OBJS = $(patsubst src/%.cu,$(BINDIR)/%.o,$(CU_SRCS))
CPP_OBJS = $(patsubst src/%.cpp,$(BINDIR)/%.o,$(CPP_SRCS))

OBJS = $(CU_OBJS) $(CPP_OBJS)

all: $(TARGET) $(UTILS)

$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(NVFLAGS) $(INCLUDES) -o $@ $^

$(BINDIR)/%.o: src/%.cu | $(BINDIR)
	$(NVCC) $(CFLAGS) $(NVFLAGS) $(INCLUDES) -c $< -o $@

$(BINDIR)/%.o: src/%.cpp | $(BINDIR)
	$(CXX) $(CFLAGS) $(NVFLAGS) $(INCLUDES) -c $< -o $@

convert: utils/convert.cpp
	$(CXX) $(CFLAGS) -o $@ $<

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(BINDIR) $(TARGET) $(UTILS)
