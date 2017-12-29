import CUDAdrv
import CUFFT
using Base.Test

dev = CuDevice(0)
ctx = CuContext(dev)

@testset begin

    # A simple 1d transform
    n = 64
    nc = div(n,2)+1
    # Create an array and move to the GPU
    a = randn(n)
    g = CUDAdrv.CuArray(a)
    # Allocate space for the FFT
    gfft = CUDAdrv.CuArray(Complex128, nc)
    # Perform the FFT
    pl! = CUFFT.plan(gfft, g)
    pl!(gfft, g, true)
    afftg = Array(gfft)
    afft = rfft(a)
    @test afftg ≈ afft
    # Perform the IFFT
    pli! = CUFFT.plan(g, gfft)
    pli!(g, gfft, false)
    # Move back to host and compare
    a2 = Array(g)
    @test a ≈ a2/n

    # 2D transform with CudaArrays
    A = rand(7,6)
    G = CUDAdrv.CuArray(A)
    GFFT = CUDAdrv.CuArray(Complex{eltype(A)}, div(size(G,1),2)+1, size(G,2))
    pl! = CUFFT.plan(GFFT, G)
    pl!(GFFT, G, true)
    AFFTG = Array(GFFT)
    AFFT = rfft(A)
    @test AFFTG ≈ AFFT
    pli! = CUFFT.plan(G,GFFT)
    pli!(G, GFFT, false)
    A2 = Array(G)
    @test A ≈ A2/length(A)

    # 2D transform with CudaPitchedArrays
    A = rand(8,3)
    G = CUDArt.CudaPitchedArray(A)
    GFFT = CUDArt.CudaPitchedArray(Complex{eltype(A)}, div(size(G,1),2)+1, size(G,2))
    pl! = CUFFT.plan(GFFT, G)
    pl!(GFFT, G, true)
    AFFTG = Array(GFFT)
    AFFT = rfft(A)
    @test AFFTG ≈ AFFT
    pli! = CUFFT.plan(G,GFFT)
    pli!(G, GFFT, false)
    A2 = Array(G)
    @test A ≈ A2/length(A)

    # 2D in-place transform
    A = rand(8,3)
    G, GFFT = CUFFT.RCpair(A)
    pl! = CUFFT.plan(GFFT, G)
    pl!(GFFT, G, true)
    AFFTG = Array(GFFT)
    AFFT = rfft(A)
    @test AFFTG ≈ AFFT
    pli! = CUFFT.plan(G,GFFT)
    pli!(G, GFFT, false)
    A2 = Array(G)
    @test A ≈ A2/length(A)

    # 3D transform using CudaArray
    NX, NY, NZ = 38, 69, 108
    A = randn(NX, NY, NZ)
    G = CUDAdrv.CuArray(A)
    GFFT = CUDAdrv.CuArray(Complex{eltype(G)}, div(size(G,1),2)+1, size(G,2), size(G,3))
    pl! = CUFFT.plan(GFFT, G)
    pl!(GFFT, G, true)
    AFFTG = Array(GFFT)
    AFFT = rfft(A)
    @test AFFTG ≈ AFFT
    pli! = CUFFT.plan(G, GFFT)
    pli!(G, GFFT, false)
    A2 = Array(G)
    @test A ≈ A2/length(A)

    # # ... and with CudaPitchedArrays
    # G = CUDArt.CudaPitchedArray(A)
    # GFFT = CUDArt.CudaPitchedArray(Complex{eltype(G)}, div(size(G,1),2)+1, size(G,2), size(G,3))
    # pl! = CUFFT.plan(GFFT, G)
    # pl!(GFFT, G, true)
    # AFFTG = Array(GFFT)
    # AFFT = rfft(A)
    # @test AFFTG ≈ AFFT
    # pli! = CUFFT.plan(G, GFFT)
    # pli!(G, GFFT, false)
    # A2 = Array(G)
    # @test A ≈ A2/length(A)

end

destroy!(ctx)
