import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from cuda_src import CUDA_SRC_PCR_THOMAS
import pycuda.driver as drv

from pycuda.compiler import SourceModule

CUDA_SRC = r"""
extern "C"
__global__ void crpcrKernel(double *d_a, 
                            double *d_b, 
                            double *d_c, 
                            double *d_d, 
                            double *d_x, 
                            unsigned int systemSizeOriginal,
                            unsigned int iterations)
{
    const unsigned int thid = threadIdx.x;
    const unsigned int blid = blockIdx.x;
    const unsigned int systemSize = blockDim.x * 2;
    const unsigned int restSystemSize = blockDim.x;
    
    extern __shared__ char shared[];

    double* a = (double*)shared;
    double* b = (double*)&a[systemSize+1];
    double* c = (double*)&b[systemSize+1];
    double* d = (double*)&c[systemSize+1];
    double* x = (double*)&d[systemSize+1];

    a[thid] = d_a[thid + blid * systemSizeOriginal];
    b[thid] = d_b[thid + blid * systemSizeOriginal];
    c[thid] = d_c[thid + blid * systemSizeOriginal];
    d[thid] = d_d[thid + blid * systemSizeOriginal];
    
    if(thid < (systemSizeOriginal - systemSize/2))
    {
        d[thid + blockDim.x] = d_d[thid + blockDim.x + blid * systemSizeOriginal];
        b[thid + blockDim.x] = d_b[thid + blockDim.x + blid * systemSizeOriginal];
        c[thid + blockDim.x] = d_c[thid + blockDim.x + blid * systemSizeOriginal];
        a[thid + blockDim.x] = d_a[thid + blockDim.x + blid * systemSizeOriginal];
    }
    else
    {
        d[thid + blockDim.x] = 0;
        b[thid + blockDim.x] = 1;
        c[thid + blockDim.x] = 0;
        a[thid + blockDim.x] = 1;    
    }
    __syncthreads();
      
    int i = 2 * thid + 1;
    if(i == systemSize - 1)
    {
        double tmp = a[i] / b[i-1];
        b[i] = b[i] - c[i-1] * tmp;
        d[i] = d[i] - d[i-1] * tmp;
        a[i] = -a[i-1] * tmp;
        c[i] = 0;
    }
    else
    {
        double tmp1 = a[i] / b[i-1];
        double tmp2 = c[i] / b[i+1];
        b[i] = b[i] - c[i-1] * tmp1 - a[i+1] * tmp2;
        d[i] = d[i] - d[i-1] * tmp1 - d[i+1] * tmp2;
        a[i] = -a[i-1] * tmp1;
        c[i] = -c[i+1] * tmp2;
    }
    
    __syncthreads();    
    
    double* aa = (double*)&x[systemSize+1];
    double* bb = (double*)&aa[restSystemSize];
    double* cc = (double*)&bb[restSystemSize];
    double* dd = (double*)&cc[restSystemSize];
    double* xx = (double*)&dd[restSystemSize];

    
    aa[thid] = a[thid*2+1];
    bb[thid] = b[thid*2+1];
    cc[thid] = c[thid*2+1];
    dd[thid] = d[thid*2+1];

    __syncthreads();

    double aNew, bNew, cNew, dNew;
    int delta = 1;

    //parallel cyclic reduction
    for (unsigned int j = 0; j < iterations; j++)
    {
        int i = thid;
        if(i < delta)
        {
            double tmp2 = cc[i] / bb[i+delta];
            bNew = bb[i] - aa[i+delta] * tmp2;
            dNew = dd[i] - dd[i+delta] * tmp2;
            aNew = 0;
            cNew = -cc[i+delta] * tmp2;
        }
        else if((restSystemSize-i-1) < delta)
        {
            double tmp = aa[i] / bb[i-delta];
            bNew = bb[i] - cc[i-delta] * tmp;
            dNew = dd[i] - dd[i-delta] * tmp;
            aNew = -aa[i-delta] * tmp;
            cNew = 0;
        }
        else
        {
            double tmp1 = aa[i] / bb[i-delta];
            double tmp2 = cc[i] / bb[i+delta];
            bNew = bb[i] - cc[i-delta] * tmp1 - aa[i+delta] * tmp2;
            dNew = dd[i] - dd[i-delta] * tmp1 - dd[i+delta] * tmp2;
            aNew = -aa[i-delta] * tmp1;
            cNew = -cc[i+delta] * tmp2;
        }
        __syncthreads();

        bb[i] = bNew;
        dd[i] = dNew;
        aa[i] = aNew;
        cc[i] = cNew;

        delta *=2;
        
        __syncthreads();
    }

    if (thid < delta)
    {
        int addr1 = thid;
        int addr2 = thid+delta;
        double tmp3 = bb[addr2]*bb[addr1]-cc[addr1]*aa[addr2];
        xx[addr1] = (bb[addr2]*dd[addr1]-cc[addr1]*dd[addr2])/tmp3;
        xx[addr2] = (dd[addr2]*bb[addr1]-dd[addr1]*aa[addr2])/tmp3;
    }
    
    __syncthreads(); 
    
    x[thid*2+1]=xx[thid];

    __syncthreads();
  
    //backward substitution    
    i = 2 * thid;
    if(i == 0)
        x[i] = (d[i] - c[i]*x[i+1]) / b[i];
    else
        x[i] = (d[i] - a[i]*x[i-1] - c[i]*x[i+1]) / b[i];
    
    __syncthreads();    

    d_x[thid + blid * systemSizeOriginal] = x[thid];
    
    if(thid < (systemSizeOriginal - systemSize/2))
        d_x[thid + blockDim.x + blid * systemSizeOriginal] = x[thid + blockDim.x];
}

"""

mod = SourceModule(CUDA_SRC_PCR_THOMAS, options=["-arch=sm_86"])

crpcr = mod.get_function("pcr_thomas")


def periodical_sweep_gpu(vec, f, r, N):
    global crpcr
    # assert (N & (N - 1)) == 0, "N-1 должно быть степенью двойки"
    systemSize = N - 1

    c = 1 + r * vec[1:]
    c = np.concatenate([c, [1 + r * vec[0]]])

    b = np.ones(N) * 4

    a = 1 - r * vec[:-1]
    a = np.concatenate([[1 - r * vec[-1]], a])

    v_n_1 = np.concatenate(([c[-1]], np.zeros(N - 3), [a[-1]]))
    u_n_1 = np.concatenate(([a[0]], np.zeros(N - 3), [c[N - 2]]))

    c_n = b[-1]

    a_sub = np.concatenate(([0], a[1:-1]))
    b_sub = b[:-1]
    c_sub = np.concatenate((c[:-2], [0]))

    a_sub = a_sub.astype(np.float64)
    b_sub = b_sub.astype(np.float64)
    c_sub = c_sub.astype(np.float64)
    f = f.astype(np.float64)
    u_n_1 = u_n_1.astype(np.float64)

    a_gpu = cuda.mem_alloc(systemSize * 8)
    b_gpu = cuda.mem_alloc(systemSize * 8)
    c_gpu = cuda.mem_alloc(systemSize * 8)
    f_gpu = cuda.mem_alloc(systemSize * 8)
    p_gpu = cuda.mem_alloc(systemSize * 8)
    u_gpu = cuda.mem_alloc(systemSize * 8)
    q_gpu = cuda.mem_alloc(systemSize * 8)

    cuda.memcpy_htod(a_gpu, a_sub)
    cuda.memcpy_htod(b_gpu, b_sub)
    cuda.memcpy_htod(c_gpu, c_sub)
    cuda.memcpy_htod(f_gpu, f[:-1])
    cuda.memcpy_htod(u_gpu, u_n_1)

    target_group_size = 8
    iterations = int(np.log2(systemSize / target_group_size))  # int(np.log2(systemSize // 2))
    shared = (5 * (systemSize + 1) + 5 * (systemSize // 2)) * 8


    group_size = N // (2 ** iterations)
    num_groups = 2 ** iterations

    # print(f"shared: {shared}\niterations:{iterations}\ngroup size: {group_size}\nnum_groups: {num_groups}")
    # exit(0)

    crpcr(a_gpu, b_gpu, c_gpu, f_gpu, p_gpu,
          np.uint32(systemSize),
          np.uint32(iterations),
          grid=(num_groups, 1, 1),
          block=(group_size, 1, 1),
          shared=shared)

    cuda.memcpy_htod(a_gpu, a_sub)
    cuda.memcpy_htod(b_gpu, b_sub)
    cuda.memcpy_htod(c_gpu, c_sub)

    crpcr(a_gpu, b_gpu, c_gpu, u_gpu, q_gpu,
          np.uint32(systemSize),
          np.uint32(iterations),
          grid=(num_groups, 1, 1),
          block=(group_size, 1, 1),  # blockDim.x = N/2 - в оригинале да, размер системы // 2
          shared=shared)

    p = np.zeros(N - 1, dtype=np.float64)
    q = np.zeros(N - 1, dtype=np.float64)

    cuda.memcpy_dtoh(p, p_gpu)
    cuda.memcpy_dtoh(q, q_gpu)

    xn = (f[-1] - v_n_1 @ p) / (c_n - v_n_1 @ q)

    x_n_1 = p - q * xn

    a_gpu.free()
    b_gpu.free()
    c_gpu.free()
    f_gpu.free()
    p_gpu.free()
    u_gpu.free()
    q_gpu.free()

    return np.concatenate((x_n_1, [xn], [x_n_1[0]]))
