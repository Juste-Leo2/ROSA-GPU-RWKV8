# examples/test_rosa_extension.py
import torch
import time

try:
    # Import the pre-compiled CUDA extension
    # Python will find the .pyd file in the same directory.
    import rosa_cuda_ext
    print("Module 'rosa_cuda_ext' imported successfully!")
except ImportError:
    print("ERROR: The 'rosa_cuda_ext' module was not found.")
    print("Please run the 'python build.py' script from the project root first.")
    exit()

# --- Step 1: Original Python implementation for comparison ---

def rosa_python(x):
    n=len(x); y=[-1]*n; s=2*n+1; b=[None]*s; c=[-1]*s; d=[0]*s; e=[-1]*s; b[0]={}; g=0; z=1
    for i,t in enumerate(x):
        r=z; z+=1; b[r]={}; d[r]=d[g]+1; p=g
        while p!=-1 and t not in b[p]: b[p][t]=r; p=c[p]
        if p==-1: c[r]=0
        else:
            q=b[p][t]
            if d[p]+1==d[q]: c[r]=q
            else:
                u=z; z+=1; b[u]=b[q].copy(); d[u]=d[p]+1; c[u]=c[q]; e[u]=e[q]
                while p!=-1 and b[p][t]==q: b[p][t]=u; p=c[p]
                c[q]=c[r]=u
        v=g=r; a=-1
        while v!=-1:
            if d[v]>0 and e[v]>=0 and e[v]+1 < len(x):
                a=x[e[v]+1]
                break
            v=c[v]
        y[i]=a; v=g
        while v!=-1 and e[v]<i: e[v]=i; v=c[v]
    return y

def rosa_batch_python(z: torch.Tensor) -> torch.Tensor:
    assert z.dtype==torch.long and z.ndim==2
    zc = z.detach().contiguous().cpu()
    results = []
    for r in zc:
        list_r = r.tolist()
        results.append(torch.as_tensor(rosa_python(list_r), dtype=z.dtype))
    return torch.stack(results).to(z.device)


# --- Step 2: Validation and performance test ---

def run_test():
    """
    Runs correctness and performance tests comparing the Python and CUDA implementations.
    """
    device = 'cuda'
    if not torch.cuda.is_available():
        print("CUDA is not available. Test cancelled.")
        return

    # Test parameters
    V, B, T = 11, 128, 128  # Vocab_size, Batch_size, Sequence_length
    
    # Generate random test data
    test_input = torch.randint(0, V, (B, T), device=device, dtype=torch.long)
    
    print("\n--- Correctness Test ---")
    
    # Execute Python implementation
    output_python = rosa_batch_python(test_input)
    
    # Execute CUDA extension
    output_cuda = rosa_cuda_ext.forward(test_input)
    
    # Compare results
    are_equal = torch.equal(output_python.cpu(), output_cuda.cpu())
    
    if are_equal:
        print("✅ SUCCESS: Python and CUDA outputs are identical!")
    else:
        print("❌ FAILURE: Outputs are different!")
        diff = (output_python.cpu() != output_cuda.cpu())
        diff_indices = torch.nonzero(diff)
        print(f"Number of different values: {diff.sum().item()}")
        if diff.sum().item() < 10: # Print details only for a few differences
            for i in range(diff_indices.shape[0]):
                b_idx, t_idx = diff_indices[i].tolist()
                py_val = output_python[b_idx, t_idx].item()
                cu_val = output_cuda[b_idx, t_idx].item()
                print(f"  - Diff at (batch={b_idx}, time={t_idx}): Python={py_val}, CUDA={cu_val}")

    print("\n--- Performance Test ---")
    
    n_runs = 100
    
    # Python Benchmark
    torch.cuda.synchronize()
    start_time_py = time.time()
    for _ in range(n_runs):
        _ = rosa_batch_python(test_input)
    torch.cuda.synchronize()
    end_time_py = time.time()
    
    python_time = (end_time_py - start_time_py) / n_runs
    print(f"Average Python time: {python_time * 1000:.3f} ms")
    
    # CUDA Benchmark
    torch.cuda.synchronize()
    start_time_cuda = time.time()
    for _ in range(n_runs):
        _ = rosa_cuda_ext.forward(test_input)
    torch.cuda.synchronize()
    end_time_cuda = time.time()

    cuda_time = (end_time_cuda - start_time_cuda) / n_runs
    print(f"Average CUDA time : {cuda_time * 1000:.3f} ms")
    
    if cuda_time > 0:
        print(f"\n🚀 Speedup: {python_time / cuda_time:.2f}x")

if __name__ == '__main__':
    run_test()