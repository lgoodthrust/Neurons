# Project Brain

An experimental, biologically inspired neural simulator focused on biological fidelity, computational efficiency, and real-time performance.

if you want "all" the details read the "WHAT_IS_THIS.txt" file.

Built from scratch in C++17 and CUDA without existing neural simulation frameworks such as NEST, Brian2, or GeNN.

## Neuron Model

- Adaptive exponential integrate-and-fire (AdEx) neurons with four dendritic compartments and back-propagating spikes.
- Conductance-based AMPA, NMDA, and GABA synapses.
- 160 explicit synaptic inputs per neuron, with Poisson background drive representing approximately 6,900 total inputs.
- Configurable time step of 0.0625–0.5 ms (default 0.1 ms).
- Axonal delays of 1–3 ms.

## GPU Architecture

- Persistent cooperative CUDA kernel with grid-wide synchronization.
- Register-based integration between minimum-delay communication epochs.
- Multiple neurons per thread, scaling with available GPU memory.
- Stateless, counter-based Philox random number generation.
- Hybrid push–pull spike delivery with warp-pooled processing.
- Compressed CSR connectivity, memory prefetching, and coalesced memory access.

## Plasticity

- Spike-timing-dependent plasticity (STDP).
- Three-factor dopamine-based reward learning.
- Timestamp-based, lazily decayed synaptic eligibility traces.

## Validation and Profiling

- Independent double-precision CPU reference.
- GPU validation against expected float32 rounding error.
- Built-in validator self-test.
- Per-phase performance measurements and Nsight Compute profiling support.

## Performance

Benchmarked on an NVIDIA RTX 5060 Ti.

| Neurons | Synapses | Simulation Speed |
|---:|---:|---:|
| 10,000 | 1.6M | 20× real time |
| 30,000 | 4.8M | 9.8× real time |
| 100,000 | 16M | 2.5× real time |
| 1,000,000 | 160M | 0.2× real time |

## Development

An independent passion project exploring alternatives to traditional AI architectures through biologically inspired computation.
Developed with AI assistance (Ironic, I know, right? Crazy.)

Technical feedback, criticism, and contributions are welcome.

## Building

Requirements:
- Visual Studio 2022 with C++ development tools
- NVIDIA CUDA Toolkit 13.5
- Windows 10 SDK
- NVIDIA GPU with CUDA compute capability 8.0 or higher (Ampere or newer). Currently tested on the RTX 5060 Ti.
- vcpkg with GLFW3 and GLEW (x64-windows-static-md)

Open `Brain.sln` in Visual Studio, select `Release | x64`, and build the solution.

Originally developed and tested on an NVIDIA RTX 5060 Ti.

## License

MIT License. See `LICENSE` for details.
