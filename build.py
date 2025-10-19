# build.py
import os
import shutil
from torch.utils.cpp_extension import load

def build_extension():
    """
    Compiles the CUDA extension using PyTorch's JIT compiler.

    This script creates a temporary 'build_cache' directory to compile the sources,
    then copies the resulting shared library (.pyd on Windows) to both the project
    root and the 'examples/' directory so it can be easily imported.
    """
    ext_name = 'rosa_cuda_ext'
    build_dir = './build_cache'
    example_dir = './examples'

    # Ensure the build and example directories exist
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(example_dir, exist_ok=True)
    
    # Clean up any previously compiled libraries in the destination folders
    for folder in ['.', example_dir]:
        for f in os.listdir(folder):
            if f.startswith(ext_name) and f.endswith('.pyd'):
                print(f"Removing old library in '{folder}': {f}")
                os.remove(os.path.join(folder, f))

    print(f"Building CUDA extension '{ext_name}'...")
    try:
        # Use PyTorch's JIT compiler to build the extension
        load(
            name=ext_name,
            sources=['rosa_cuda.cpp', 'rosa_cuda_kernel.cu'],
            verbose=True,
            build_directory=build_dir
        )
        print("Build successful!")
    except Exception as e:
        print(f"ERROR: Build failed: {e}")
        return

    # Find the compiled library file in the build directory
    built_lib_path = None
    for file in os.listdir(build_dir):
        if file.startswith(ext_name) and file.endswith('.pyd'):
            built_lib_path = os.path.join(build_dir, file)
            break

    if built_lib_path:
        # Copy the compiled library to the project root and the examples folder
        lib_filename = os.path.basename(built_lib_path)
        dest_paths = [
            os.path.join('.', lib_filename),
            os.path.join(example_dir, lib_filename)
        ]
        
        for dest in dest_paths:
            print(f"Copying '{built_lib_path}' to '{dest}'")
            shutil.copy(built_lib_path, dest)
        
        print("-" * 60)
        print(f"✅ SUCCESS: The '{ext_name}' extension is ready.")
        print("You can now import it in your Python scripts via 'import rosa_cuda_ext'")
        print("-" * 60)
    else:
        print("ERROR: Could not find the compiled .pyd file in the build directory.")
        
    print(f"Build cache has been preserved in '{build_dir}'.")

if __name__ == '__main__':
    build_extension()