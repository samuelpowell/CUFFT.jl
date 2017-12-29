# CUFFT

This is a fork of [CUFFT.jl](https://github.com/JuliaGPU/CUFFT.jl) which uses [CUDAdrv.jl](https://github.com/JuliaGPU/CUDAdrv.jl) as a backend. This fork does not support pitched pointers, and related functionality is disabled.

## Usage example

Here's an example of taking a 2D real transform, and then it's inverse, and comparing against Julia's CPU-based 

```julia
using CUDAdrv, CUFFT, Base.Test

ctx = CuContext(CuDevice(0))
A = rand(7,6)
G = CuArray(A)
GFFT = CuArray{Complex{eltype(A)}}(div(size(G,1),2)+1, size(G,2))
pl! = plan(GFFT, G)
pl!(GFFT, G, true)
AFFTG = Array(GFFT)
AFFT = rfft(A)
@test AFFTG ≈ AFFT
pli! = plan(G,GFFT)
pli!(G, GFFT, false)
A2 = Array(G)
@test A ≈ A2/length(A)
```
