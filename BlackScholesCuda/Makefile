include Makefile.include


HW1_INCS = black_scholes.h parser.h timer.h util.h
HW1_C_SRCS = main.c parser.c
HW1_CUDA_SRCS = black_scholes.cu
HW1_C_OBJS = $(HW1_C_SRCS:.c=.o)
HW1_CUDA_OBJS = $(HW1_CUDA_SRCS:.cu=.o)
HW1_EXE = hw1.x

all: hw1.x

%.o: %.c
	$(CC) -c $(CCFLAGS) $(ACCFLAGS) $< -o $@

%.o: %.cu
	$(CC) -c $(CCFLAGS) $(ACCFLAGS) $< -o $@

hw1.x: $(HW1_C_OBJS) $(HW1_CUDA_OBJS)
	$(CC) $(LDFLAGS) $(HW1_C_OBJS) $(HW1_CUDA_OBJS) -o $@

black_scholes.o: black_scholes.cu black_scholes.h util.h
main.o: main.c black_scholes.h parser.h
parser.o: parser.c parser.h
util.o: util.c util.h

clean:
	rm -f $(HW1_C_OBJS) $(HW1_CUDA_OBJS) $(HW1_EXE)