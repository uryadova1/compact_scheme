

CUDA_SRC_PCR_THOMAS = r"""
extern "C"
__global__ void pcr_thomas(double *d_a, 
                            double *d_b, 
                            double *d_c, 
                            double *d_d, 
                            double *d_x, 
                            unsigned int sizeSystem,
                            unsigned int iterations){
                            
    const unsigned int tid = threadIdx.x;           // Номер потока (0,1,2,...)
    const unsigned int bid = blockIdx.x;            // Номер системы (если их много)
    
    // Размер группы после PCR
    const unsigned int group_size = sizeSystem / (1 << iterations);
    
    // Шаг (stride) между уравнениями в группе - это кажется аналог delta из статьи
    const unsigned int stride = 1 << iterations;
    
    extern __shared__ char shared[];  // Общая память для блока
    
    // 5 массивов для хранения уравнений группы
    double *a = (double*)shared;                          // Нижняя диагональ
    double *b = (double*)&a[group_size];                 // Главная диагональ
    double *c = (double*)&b[group_size];                 // Верхняя диагональ
    double *d = (double*)&c[group_size];                 // Правая часть
    double *x = (double*)&d[group_size];                 // Решение
    
    double *a_pcr = a;   // Переиспользуем те же массивы - для записи массивов для томаса
    double *b_pcr = b;
    double *c_pcr = c;
    double *d_pcr = d;
    
    if (tid < group_size) {
        // Вычисляем глобальный индекс уравнения
        // Уравнения в группе идут с шагом stride
        int global_idx = bid * stride + tid * stride;
        
        // Загружаем диагонали и правую часть
        a[tid] = d_a[global_idx];
        b[tid] = d_b[global_idx];
        c[tid] = d_c[global_idx];
        d[tid] = d_d[global_idx];
        
        // Инициализация решения
        x[tid] = 0.0;
    }
    __syncthreads();
    
    double aNew, bNew, cNew, dNew;
    int delta = 1;
    
    for (int j = 0; j <iterations; j++)
    {
        int i = threadIdx.x;
        int iRight = i + delta;
        if (iRight >= sizeSystem) iRight = sizeSystem - 1;
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
        delta <<= 1;
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
    
    thomas(a, b, c, d, x, group_size, tid);
    __syncthreads();
    
    // Сохранение результата
    if (tid < group_size) {
        int idx = bid * stride + tid * stride;
        d_x[idx] = x[tid];
    }                                                  
}

__device__ void thomas(double *a, double *b, double *c, double *d, double *x, 
                       int group_size, int tid)
{
    // В Thomas алгоритме из статьи используется stride
    // Но для независимой группы stride = 1
    const int stride = 1;
    
    // Forward elimination - только один поток выполняет (как в статье)
    if (tid == 0) {
        // Первое уравнение
        c[0] = c[0] / b[0];
        d[0] = d[0] / b[0];
        
        // Остальные уравнения
        for (int i = 1; i < group_size; i++) {
            double tmp = b[i] - a[i] * c[i - stride];
            c[i] = c[i] / tmp;
            d[i] = (d[i] - d[i - stride] * a[i]) / tmp;
        }
        
        // Backward substitution
        x[group_size - 1] = d[group_size - 1];
        for (int i = group_size - 2; i >= 0; i--) {
            x[i] = d[i] - c[i] * x[i + stride];
        }
    }
}
"""