#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

extern "C" void launch_rosa_kernel(
    const int64_t* x_ptr,
    int64_t* y_ptr,
    int64_t* b_ptr,
    int64_t* c_ptr,
    int64_t* d_ptr,
    int64_t* e_ptr,
    int B, int T, int V);

void rosa_forward_cuda(
    const torch::Tensor& x,
    torch::Tensor& y,
    int V)
{
    const int B = x.size(0);
    const int T = x.size(1);
    const int max_states = 2 * T + 1;


    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kInt64);

    auto b = torch::full({B, max_states, V}, -1, options);
    auto c = torch::full({B, max_states}, -1, options);
    auto d = torch::zeros({B, max_states}, options);
    auto e = torch::full({B, max_states}, -1, options);

    launch_rosa_kernel(
        x.data_ptr<int64_t>(),
        y.data_ptr<int64_t>(),
        b.data_ptr<int64_t>(),
        c.data_ptr<int64_t>(),
        d.data_ptr<int64_t>(),
        e.data_ptr<int64_t>(),
        B, T, V);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor rosa_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kInt64, "Input tensor must be of type torch.long");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be a 2D tensor");

    int V = x.max().item<int64_t>() + 1;

    auto y = torch::empty_like(x);

    rosa_forward_cuda(x, y, V);

    return y;
}

PYBIND11_MODULE(rosa_cuda_ext, m) { 
    m.def("forward", &rosa_forward, "ROSA forward (CUDA)");
}