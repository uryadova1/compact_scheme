CUDA_SRC_PCR_THOMAS = r"""
extern "C"

__device__ void thomas(double *a, double *b, double *c, double *d, double *x, 
                       int group_size, int tid)
{   
    int stride = 1;
    c[tid] = c[tid] / b[tid];
    d[tid] = d[tid] / b[tid];
    int startLocationSystem = stride + tid; //что такое страйд, мб я просто не те ур-я считаю
    for (int i = startLocationSystem; i < group_size; i += stride)
    {
    double tmp = (b[i] - a[i] * c[i - stride]);
    c[i] = c[i] / tmp;
    d[i] = (d[i] - d[i - stride] * a[i]) / tmp;
    }
    
    int endLocationSystem = group_size - stride + tid;
    x[endLocationSystem] = d[endLocationSystem];
    
    for (int i = endLocationSystem - stride; i >= 0; i -= stride)
    {
        x[i] = d[i] - c[i] * x[i + stride];
    }
}

__global__ void pcr_thomas(double *d_a, 
                            double *d_b, 
                            double *d_c, 
                            double *d_d, 
                            double *d_x, 
                            unsigned int sizeSystem,
                            unsigned int iterations){
                            
    const unsigned int tid = threadIdx.x;           
    const unsigned int bid = blockIdx.x;           
    
    // Размер группы после PCR
    const unsigned int group_size = sizeSystem / (1 << iterations);
    
    // Шаг (stride) между уравнениями в группе 
    const unsigned int stride = 1 << iterations;
    
    //if (tid == 0 && bid == 0) printf("%u", group_size);

    
    extern __shared__ char shared[];  
    
    double *a = (double*)shared;                          // Нижняя диагональ
    double *b = a + sizeSystem;                 // Главная диагональ
    double *c = b + sizeSystem;                 // Верхняя диагональ
    double *d = c + sizeSystem;                 // Правая часть
    double *x = d + sizeSystem;                 // Решение
    
    a[tid] = d_a[tid];
    b[tid] = d_b[tid];
    c[tid] = d_c[tid];
    d[tid] = d_d[tid];
    
    // Инициализация решения
    x[tid] = 0.0;
    
    __syncthreads();
    
    double aNew, bNew, cNew, dNew;
    int delta = 1;
    int i = tid;
    for (int j = 0; j < iterations; ++j)
    {
        int iRight = i + delta;
        if (iRight >= group_size) iRight = group_size - 1; //я не понимаю размер группы или системы
        int iLeft = i - delta;
        if (iLeft < 0) iLeft = 0;
        
        double tmp1 = a[i] / b[iLeft];
        double tmp2 = c[i] / b[iRight];
        bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
        dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
        aNew = -a[iLeft] * tmp1;
        cNew = -c[iRight] * tmp2;
        
        __syncthreads();
        b[i] = bNew;
        d[i] = dNew;
        a[i] = aNew;
        c[i] = cNew;
        __syncthreads();
        delta <<= 1;
    }
    
    if (tid < delta)
    {
        int addr1 = tid;
        int addr2 = tid + delta;

        float tmp = b[addr2] * b[addr1] - c[addr1] * a[addr2];
        x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp;
        x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp;
    }
    
    __syncthreads();
    
    thomas(a, b, c, d, x, group_size, tid);
    __syncthreads();
    
    // Сохранение результата
    if (tid < group_size) {
        int idx = bid * stride + tid * stride;
        d_x[idx] = x[tid];
    }                                                  
}


"""


