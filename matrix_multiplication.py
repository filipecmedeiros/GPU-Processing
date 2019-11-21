from __future__ import division
from numba import cuda
import numpy
import math
import time


# CUDA kernel
@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp
        
size = 1000
# Host code
start_time = time.time()

# Initialize the data array
#A = numpy.random.randint(0, 10, size=(size,size))
#B = numpy.random.randint(0, 10, size=(size,size))

A = numpy.full((size, size), 10, numpy.float) # matrix containing all 3's
B = numpy.full((size, size), 10, numpy.float) # matrix containing all 4's

# Copy the arrays to the device
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)


# Allocate memory on the device for the result
C_global_mem = cuda.device_array((size, size))

# Configure the blocks
threadsperblock = (16, 16)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Start the kernel 
matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)

# Copy the result back to the host
C = C_global_mem.copy_to_host()

#print(C)
print("--- %s seconds ---" % (time.time() - start_time))