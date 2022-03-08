# MCGPULite

A Lightweight version of [MC-GPU v1.3](https://github.com/DIDSR/MCGPU/) (A GPU-Accelerated Monte Carlo X-Ray Transport Simulation Tool).

## What's Different

- [Makefile](https://github.com/z0gSh1u/MCGPULite/blob/master/Makefile) is rewritten to support standalone `zlib`, `CUDA`, `nvcc` and `openmpi` so that `sudo apt-get` is no more needed.

  - Have CUDA installed correctly.

  - Download zlib and openmpi. Compile them.

    ```sh
    # zlib
    ./configure
    make
    make install prefix=/zlib/install/path
    # openmpi
    ./configure --prefix=/openmpi/install/path
    make
    make install
    ```

  - Modify paths in Makefile according to your machine.

    ```makefile
    # You should modify these according to your machine.
    NVCC = /usr/local/cuda-10.1/bin/nvcc
    CUDA_INCLUDE = /usr/local/cuda-10.1/include/
    CUDA_LIB = /usr/local/cuda-10.1/lib64/
    CUDA_SDK_INCLUDE = /usr/local/cuda-10.0/samples/common/inc/
    CUDA_SDK_LIB = /usr/local/cuda-10.0/samples/common/lib/linux/x86_64/
    OPENMPI_INCLUDE = /home/zhuoxu/app/openmpi-4.1.1/install/include/
    OPENMPI_LIB = /home/zhuoxu/app/openmpi-4.1.1/install/lib/
    ZLIB_INCLUDE = /home/zhuoxu/app/zlib-1.2.11/install/include/
    ZLIB_LIB = /home/zhuoxu/app/zlib-1.2.11/install/lib/
    ```

  - Compile MCGPULite.

    ```sh
    make clean
    make
    ```
    
  - You can create a symbol link and add it to PATH if you want.

    ```sh
    ln -s MCGPULite_v1.3.x MCGPULite
    
    # vim ~/.bashrc
    export PATH="/path/to/MCGPULite/folder/:$PATH"
    # source ~/.bashrc
    ```

- Dose information is longer reported. And [input file](./MCGPULite_v1.3.sample.in) doesn't require `[SECTION DOSE DEPOSITION]` now.

  ```diff
  # [SECTION CT SCAN TRAJECTORY v.2011-10-25]
  ...
  
  - # [SECTION DOSE DEPOSITION v.2012-12-12]
  - ...
  
  # [SECTION VOXELIZED GEOMETRY FILE v.2009-11-30]
  ...
  ```

- The output raw file consists of only 3 slices: **Total, Primary, Scatter**. Individual scatters (Compton, Rayleigh, Multi Scatter) are no longer reported.

- The bug of multiple GPUs run when all GPUs are detected connected to one monitor is fixed. 

  Now you can run it like this **on one GPU**:

  ```sh
  # .in file. Specify which CUDA GPU to run on here.
  3              # GPU NUMBER TO USE WHEN MPI IS NOT USED, OR TO BE AVOIDED IN MPI RUNS
  
  # Command line. Just as it should be.
  ./MCGPULite ./<input_file>.in
  ```

  Or like this **on multiple GPUs**:

  ```sh
  # .in file. Use -1 always here.
  -1              # GPU NUMBER TO USE WHEN MPI IS NOT USED, OR TO BE AVOIDED IN MPI RUNS
  
  # Command line. Use CUDA_VISIBLE_DEVICES to specify GPUs, and pass GPU counts to -np.
  mpirun -x CUDA_VISIBLE_DEVICES=0,1,2,3 -np 4 ./MCGPULite ./<input_file>.in
  ```

  Through experiments, **I don't recommend you to use multiple GPUs if your TOTAL NUMBER OF HISTORIES is less than 1e10** because of the distribution overhead. Otherwise, I achieve a time cutdown by 66% with 1e11 histories.

- The terminal outputs talk less.

## Known Issues

- If you terminate MCGPULite with ^Z or ^C, its child process and thread are not killed. You should manually use `kill -9 <pid>` to kill.
