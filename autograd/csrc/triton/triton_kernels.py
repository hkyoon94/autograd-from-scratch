import torch
import triton
import triton.language as tl

from autograd.csrc.triton.ptxget import CompiledKernelExtractor

ext = CompiledKernelExtractor()


def add_():
    a = torch.randn(2500, dtype=torch.float32, device="cuda:0")
    b = torch.randn(2500, dtype=torch.float32, device="cuda:0")
    c = torch.empty(2500, dtype=torch.float32, device="cuda:0")
    L = 2500
    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(L, META["BLOCK_SIZE"]), )

    @ext.extract(
        a, b, c,
        L,
        BLOCK_SIZE,
        grid=grid,
    )
    @triton.jit
    def add(
        a_ptr, b_ptr, c_ptr,
        L: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        off = pid * BLOCK_SIZE
        for k in range(BLOCK_SIZE):
            mask = off + k < L
            x = tl.load(a_ptr + off + k, mask=mask)
            y = tl.load(b_ptr + off + k, mask=mask)
            tl.store(c_ptr + off + k, x + y, mask=mask)


def mm_():
    M, N, K = 1000, 500, 750
    a = torch.randn(size=(M, K), dtype=torch.float32, device="cuda:0")
    b = torch.randn(size=(K, N), dtype=torch.float32, device="cuda:0")
    c = torch.empty(size=(M, N), dtype=torch.float32, device="cuda:0")
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    @ext.extract(
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        64, 64, 32, 8,  # bs_M, bs_N, bs_K, num_M_grps
        grid=grid,
        num_warps=8,
        num_stages=2,
    )
    @triton.jit
    def mm(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(stride_am > 0)
        tl.assume(stride_ak > 0)
        tl.assume(stride_bn > 0)
        tl.assume(stride_bk > 0)
        tl.assume(stride_cm > 0)
        tl.assume(stride_cn > 0)

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(tl.float16)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


# add_()
mm_()
