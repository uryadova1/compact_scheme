CUDA_SRC_PCR_THOMAS = r"""
extern "C"
__device__ void thomas_strided(double *a, double *b, double *c, double *d, double *x, 
                                 int sizeSmallerSystem, int stride){
        // Forward elimination
    c[threadIdx.x] = c[threadIdx.x] / b[threadIdx.x];
    d[threadIdx.x] = d[threadIdx.x] / b[threadIdx.x];
    int startLocationSystem = stride + threadIdx.x;
    for (int i = startLocationSystem; i < sizeSmallerSystem; i += stride)
        {   
            double tmp = (b[i] - a[i] * c[i - stride]);
            c[i] = c[i] / tmp;
            d[i] = (d[i] - d[i - stride] * a[i]) / tmp;
        }
    // Backward substitution
    int endLocationSystem = sizeSmallerSystem - stride + threadIdx.x;
    x[endLocationSystem] = d[endLocationSystem];
    for (int i = endLocationSystem - stride; i >= 0; i -= stride)
        {   
            //if (threadIdx.x == 0) printf("i = %d\n", i);
            x[i] = d[i] - c[i] * x[i + stride];
        }                           
}

__global__ void pcr_thomas(double *d_a, 
                            double *d_b, 
                            double *d_c, 
                            double *d_d, 
                            double *d_x, 
                            unsigned int sizeSystem,
                            unsigned int iterations,
                            unsigned int stride)
{   
    const unsigned int group_size = sizeSystem / stride;
    const unsigned int tid = threadIdx.x;       
 
    const unsigned int bid = blockIdx.x;        
   // printf(">>> Kernel start: sizeSystem=%u, stride=%u, bid=%u\n, group size=%u\n", sizeSystem, stride, bid, group_size);
    
    extern __shared__ char shared[];  
    
    double *a = (double*)shared;
    double *b = a + sizeSystem;
    double *c = b + sizeSystem;
    double *d = c + sizeSystem;
    double *x = d + sizeSystem;
    
    // Загрузка данных
    if (tid < sizeSystem) {
        a[tid] = d_a[tid];
        b[tid] = d_b[tid];
        c[tid] = d_c[tid];
        d[tid] = d_d[tid];
        x[tid] = 0.0;
    }
    __syncthreads();
    
    //if (bid == 15 && tid == 63)printf("d[tid] = %f\n", d[tid]);
    
    // PCR PHASE
    double aNew, bNew, cNew, dNew;
    int delta = 1;
    for (int j = 0; j < iterations; ++j)
    {
        if (tid < sizeSystem) {
            int iRight = tid + delta;
            if (iRight >= sizeSystem) iRight = sizeSystem - 1;
            int iLeft = tid - delta;
            if (iLeft < 0) iLeft = 0;
            
            double tmp1 = a[tid] / b[iLeft];
            double tmp2 = c[tid] / b[iRight];
            bNew = b[tid] - c[iLeft] * tmp1 - a[iRight] * tmp2;
            dNew = d[tid] - d[iLeft] * tmp1 - d[iRight] * tmp2;
            aNew = -a[iLeft] * tmp1;
            cNew = -c[iRight] * tmp2;
            
            a[tid] = aNew;
            b[tid] = bNew;
            c[tid] = cNew;
            d[tid] = dNew;
        }
        __syncthreads();
        delta <<= 1;
    }
   // if (tid == 0 && bid == 0) printf(">>> Start Thomas\n"); 
    
    //if (tid % stride == 0) printf("tid = %d, bid = %d\n", tid, bid);
    if (tid < stride) {
        //int global_pos = bid + tid * stride;  // позиция первого уравнения подсистемы
        //printf("global pos = %d, tid = %d, bid = %d, d[global pos] = %f\n",global_pos, tid, bid, d[global_pos]);
        thomas_strided(a, b, c, d, x, sizeSystem, stride);
    }
    __syncthreads();
    

    d_x[tid] = x[tid];
    __syncthreads();
    if (tid == 0 && bid == 0) {
   // printf(">>> Kernel end: pos=%d, x[%d]=%f\n", bid, bid, x[bid]);
    }
}
"""
