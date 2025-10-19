#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint> 

__global__ void rosa_kernel(
    const int64_t* __restrict__ x,
    int64_t* __restrict__ y,
    int64_t* __restrict__ b,
    int64_t* __restrict__ c,
    int64_t* __restrict__ d,
    int64_t* __restrict__ e,
    int B,
    int T,
    int V)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= B) return;

    const int64_t* x_ptr = x + batch_idx * T;
    int64_t* y_ptr = y + batch_idx * T;
    
    int max_states = 2 * T + 1;
    int64_t* b_ptr = b + batch_idx * max_states * V;
    int64_t* c_ptr = c + batch_idx * max_states;
    int64_t* d_ptr = d + batch_idx * max_states;
    int64_t* e_ptr = e + batch_idx * max_states;

    int64_t g = 0;
    int64_t z = 1;

    for (int i = 0; i < T; ++i) {
        int64_t t = x_ptr[i];
        
        int64_t r = z++;
        d_ptr[r] = d_ptr[g] + 1;
        int64_t p = g;

        while (p != -1 && b_ptr[p * V + t] == -1) {
            b_ptr[p * V + t] = r;
            p = c_ptr[p];
        }

        if (p == -1) {
            c_ptr[r] = 0;
        } else {
            int64_t q = b_ptr[p * V + t];
            if (d_ptr[p] + 1 == d_ptr[q]) {
                c_ptr[r] = q;
            } else {
                int64_t u = z++;
                for (int tok = 0; tok < V; ++tok) {
                    b_ptr[u * V + tok] = b_ptr[q * V + tok];
                }
                d_ptr[u] = d_ptr[p] + 1;
                c_ptr[u] = c_ptr[q];
                e_ptr[u] = e_ptr[q];

                while (p != -1 && b_ptr[p * V + t] == q) {
                    b_ptr[p * V + t] = u;
                    p = c_ptr[p];
                }
                c_ptr[q] = u;
                c_ptr[r] = u;
            }
        }

        int64_t v = g = r;
        int64_t a = -1;

        while (v != -1) {
            if (d_ptr[v] > 0 && e_ptr[v] >= 0) {
                if (e_ptr[v] + 1 < T) {
                   a = x_ptr[e_ptr[v] + 1];
                }
                break;
            }
            v = c_ptr[v];
        }
        y_ptr[i] = a;

        v = g;
        while (v != -1 && e_ptr[v] < i) {
            e_ptr[v] = i;
            v = c_ptr[v];
        }
    }
}


extern "C" void launch_rosa_kernel(
    const int64_t* x_ptr,
    int64_t* y_ptr,
    int64_t* b_ptr,
    int64_t* c_ptr,
    int64_t* d_ptr,
    int64_t* e_ptr,
    int B, int T, int V)
{
    rosa_kernel<<<B, 1>>>(
        x_ptr, y_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
        B, T, V);
}