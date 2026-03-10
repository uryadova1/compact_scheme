import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import pycuda.driver as drv

from pycuda.compiler import SourceModule

SIMPLE_CUSA_SRC = r"""__global__ void thomas_kernel(
    double *a,
    double *b,
    double *c,
    double *d,
    double *x,
    int N)
{
    int tid = threadIdx.x;

    __shared__ double c_star[1024];
    __shared__ double d_star[1024];

    if (tid == 0)
    {
        c_star[0] = c[0] / b[0];
        d_star[0] = d[0] / b[0];

        for (int i = 1; i < N; i++)
        {
            double m = 1.0 / (b[i] - a[i] * c_star[i-1]);
            c_star[i] = c[i] * m;
            d_star[i] = (d[i] - a[i] * d_star[i-1]) * m;
        }

        x[N-1] = d_star[N-1];

        for (int i = N-2; i >= 0; i--)
        {
            x[i] = d_star[i] - c_star[i] * x[i+1];
        }
    }
}"""

CUDA_SRC = R"""template<typename T>
__device__ void thomas_kernel(
    T* a, T* b, T* c, T* d, T* x,
    int stride,
    int systemSize)
{
    int tid = threadIdx.x;

    // Forward elimination
    int first = tid;
    c[first] = c[first] / b[first];
    d[first] = d[first] / b[first];

    for (int i = first + stride; i < systemSize; i += stride)
    {
        T tmp = b[i] - a[i] * c[i - stride];
        c[i] = c[i] / tmp;
        d[i] = (d[i] - a[i] * d[i - stride]) / tmp;
    }

    // Back substitution
    int last = systemSize - stride + tid;
    x[last] = d[last];

    for (int i = last - stride; i >= 0; i -= stride)
    {
        x[i] = d[i] - c[i] * x[i + stride];
    }
}


template<typename T>
__global__ void pcr_thomas_kernel(
    T* a_g,
    T* b_g,
    T* c_g,
    T* d_g,
    T* x_g,
    int systemSize,
    int pcrSteps)
{
    extern __shared__ T smem[];

    T* a = smem;
    T* b = &a[systemSize];
    T* c = &b[systemSize];
    T* d = &c[systemSize];
    T* x = &d[systemSize];

    int tid = threadIdx.x;
    int gid = blockIdx.x * systemSize + tid;

    // загрузка системы
    if (tid < systemSize)
    {
        a[tid] = a_g[gid];
        b[tid] = b_g[gid];
        c[tid] = c_g[gid];
        d[tid] = d_g[gid];
    }

    __syncthreads();

    // ---------- PCR reduction ----------
    int delta = 1;

    for (int step = 0; step < pcrSteps; step++)
    {
        int i = tid;

        int left  = i - delta;
        int right = i + delta;

        if (left < 0) left = 0;
        if (right >= systemSize) right = systemSize - 1;

        T alpha = a[i] / b[left];
        T beta  = c[i] / b[right];

        T b_new = b[i] - c[left]*alpha - a[right]*beta;
        T d_new = d[i] - d[left]*alpha - d[right]*beta;
        T a_new = -a[left]*alpha;
        T c_new = -c[right]*beta;

        __syncthreads();

        a[i] = a_new;
        b[i] = b_new;
        c[i] = c_new;
        d[i] = d_new;

        delta *= 2;

        __syncthreads();
    }

    // ---------- Thomas solve ----------
    // после PCR каждая нить решает свою систему
    int stride = delta;

    if (tid < stride)
    {
        thomas_strided(a + tid,
                       b + tid,
                       c + tid,
                       d + tid,
                       x + tid,
                       stride,
                       systemSize);
    }

    __syncthreads();

    // запись решения
    if (tid < systemSize)
    {
        x_g[gid] = x[tid];
    }
}"""


def periodical_sweep_gpu(vec, f, r, N):
    mod = SourceModule(SIMPLE_CUSA_SRC)
    thomas_gpu = mod.get_function("thomas_kernel")

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

    a_gpu = cuda.mem_alloc(a_sub.nbytes)
    b_gpu = cuda.mem_alloc(b_sub.nbytes)
    c_gpu = cuda.mem_alloc(c_sub.nbytes)

    p_gpu = cuda.mem_alloc((N - 1) * 8)
    q_gpu = cuda.mem_alloc((N - 1) * 8)

    f_gpu = cuda.mem_alloc((N - 1) * 8)
    u_gpu = cuda.mem_alloc((N - 1) * 8)

    cuda.memcpy_htod(a_gpu, a_sub)
    cuda.memcpy_htod(b_gpu, b_sub)
    cuda.memcpy_htod(c_gpu, c_sub)

    cuda.memcpy_htod(f_gpu, f[:-1])
    cuda.memcpy_htod(u_gpu, u_n_1)

    thomas_gpu(
        a_gpu, b_gpu, c_gpu,
        f_gpu, p_gpu,
        np.int32(N - 1),
        block=(1, 1, 1), grid=(1, 1)
    )

    thomas_gpu(
        a_gpu, b_gpu, c_gpu,
        u_gpu, q_gpu,
        np.int32(N - 1),
        block=(1, 1, 1), grid=(1, 1)
    )

    p = np.empty(N - 1)
    q = np.empty(N - 1)

    cuda.memcpy_dtoh(p, p_gpu)
    cuda.memcpy_dtoh(q, q_gpu)

    xn = (f[-1] - v_n_1 @ p) / (c_n - v_n_1 @ q)

    x_n_1 = p - q * xn

    return np.concatenate((x_n_1, [xn], [x_n_1[0]]))
