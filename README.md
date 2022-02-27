# MCGPULite

A Lightweight version of MC-GPU v1.3 (A GPU-Accelerated Monte Carlo X-Ray Transport Simulation Tool).

## MCGPU Documentation

Please refer to the original [MCGPU Repository](https://github.com/DIDSR/MCGPU/) .

## What's Different

- Makefile is rewritten to support standalone zlib and nvcc.

  - Modify paths in Makefile below `# You should modify these according to your machine.`

  - Compile with

    ```sh
    make clean
    make
    ```

- Dosage information is longer reported. And [input file](./MCGPULite_v1.3.sample.in) doesn't require `[SECTION DOSE]` now.

- The output raw file consists of only 3 slices: **Total, Primary, Scatter**. Separated scatters are no longer reported.

- The terminal outputs talk less.
