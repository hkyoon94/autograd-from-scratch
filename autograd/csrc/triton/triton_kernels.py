import itertools

import torch
import triton
import triton.language as tl

from ptxget import CompiledKernelExtractor

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
        """
        'This code is from Triton official tutorial.'

        Kernel for computing the matmul C = A x B.
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


"""
def flash_attn_naive(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor
) -> torch.Tensor:
    B, H = Q.shape[0], Q.shape[1]
    q_len, kv_len = Q.shape[2], K.shape[2]
    device = Q.device

    O = torch.empty_like(Q)

    for b in range(B):
        for h in range(H):
            Tr = triton.cdiv(q_len, Br)
            Tc = triton.cdiv(kv_len, Bc)

            for i in range(Tr):
                # loading Q_i panel to SRAM
                i_off = i * Br
                Q_i = Q[b, h, i_off: i_off + Br].to(torch.float32)

                O_i_j = torch.zeros((Br, d), dtype=torch.float32, device=device)
                l_i_j = torch.zeros((Br, ), dtype=torch.float32, device=device)
                m_i_j = torch.empty((Br, ), dtype=torch.float32, device=device).fill_(-1e12)

                for j in range(Tc):  # block-tiled online-softmax matmul
                    # loading K_j, V_j to SRAM
                    j_off = j * Bc
                    K_j = K[b, h, j_off: j_off + Bc].to(torch.float32)
                    V_j = V[b, h, j_off: j_off + Bc].to(torch.float32)

                    # computing block partial score
                    S_i_j = Q_i.mm(K_j.T)  # (Br, Bc)

                    # computing partial-scaled softmax
                    m_i_j_new = torch.maximum(m_i_j, S_i_j.max(dim=-1)[0])  # (Br,)
                    P_i_j = torch.exp(S_i_j - m_i_j_new[:, None])           # (Br, Bc)
                    
                    # renormalized score-value mm
                    scale = torch.exp(m_i_j - m_i_j_new)                    # (Br,)
                    l_i_j = l_i_j.mul(scale) + P_i_j.sum(dim=-1)            # (Br,)
                    O_i_j = scale[:, None].mul(O_i_j) + P_i_j.mm(V_j)       # (Br,)

                    m_i_j = m_i_j_new

                # writing accumulated output to DRAM
                O[b, h, i_off: i_off + Br] = (1 / l_i_j)[:, None].mul(O_i_j)
    return O
"""


def get_cuda_autotune_config():
    Brs = [256, 128, 64, 32]
    Bcs = [256, 128, 64, 32]
    num_stages = [5, 4, 3]
    num_warps = [8, 4, 2]
    return list(
        triton.Config(kwargs={"Br": Br, "Bc": Bc}, num_stages=ns, num_warps=nw)
        for Br, Bc, ns, nw in itertools.product(Brs, Bcs, num_stages, num_warps)
    )


# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=["q_len", "kv_len", "d"],
# )
@triton.jit
def my_flash_attn_kernel(  # TODO: solve when param::d is not a power of 2.
    Q_ptr, K_ptr, V_ptr, O_ptr,
    q_len, kv_len,
    Q_st0, Q_st1, Q_st2, Q_st3,
    K_st0, K_st1, K_st2, K_st3,
    V_st0, V_st1, V_st2, V_st3,
    d: tl.constexpr,
    is_causal: int,
    num_kv_groups: int,
    Br: tl.constexpr,
    Bc: tl.constexpr,
):
    # Parallel grid for each batch, head
    # Parallel grid for each i * Br: (i+1) * Br segment in sequence length dimension
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)

    tl.assume(h >= 0)
    tl.assume(i >= 0)
    tl.assume(Q_st0 >= 0)
    tl.assume(Q_st1 >= 0)
    tl.assume(Q_st2 >= 0)
    tl.assume(Q_st3 >= 0)
    tl.assume(K_st0 >= 0)
    tl.assume(K_st1 >= 0)
    tl.assume(K_st2 >= 0)
    tl.assume(K_st3 >= 0)
    tl.assume(V_st0 >= 0)
    tl.assume(V_st1 >= 0)
    tl.assume(V_st2 >= 0)
    tl.assume(V_st3 >= 0)
    tl.assume(num_kv_groups >= 1)

    # loading Q_i panel to SRAM
    # same with ~[b, h, i * Br: i * Br + Br, :]
    # only using 2-dimensional broadcasting will load 4D-tensor as 2D.
    l_offs = i * Br + tl.arange(0, Br)[:, None]
    d_offs = tl.arange(0, d)[None, :]
    Q_i_ptr = Q_ptr + b * Q_st0 + h * Q_st1 + l_offs * Q_st2 + d_offs * Q_st3

    Q_i = tl.load(Q_i_ptr, mask=l_offs < q_len, other=0.0)

    # initializing accumulator tensors in SRAM
    O_i_j = tl.zeros((Br, d), dtype=tl.float32)
    l_i_j = tl.zeros((Br,), dtype=tl.float32)
    m_i_j = tl.full((Br,), -float("inf"), dtype=tl.float32)

    Tc = tl.cdiv(kv_len, Bc)
    d_inv = 1 / (d ** 0.5)

    # block-tiled online-softmax & matmul V-tile
    for j in range(Tc):
        # loading K_j, V_j to SRAM
        # same with ~[b, h, j * Bc: j * Bc + Bc, :]
        block_l_offs = j * Bc + tl.arange(0, Bc)[:, None]
        block_d_offs = tl.arange(0, d)[None, :]
        K_j_ptr = K_ptr + b * K_st0 + (h // num_kv_groups) * K_st1 + \
            block_l_offs * K_st2 + block_d_offs * K_st3
        V_j_ptr = V_ptr + b * V_st0 + (h // num_kv_groups) * V_st1 + \
            block_l_offs * V_st2 + block_d_offs * V_st3
        
        K_j = tl.load(K_j_ptr, mask=block_l_offs < kv_len, other=0.0)  # (Bc, d)
        V_j = tl.load(V_j_ptr, mask=block_l_offs < kv_len, other=0.0)  # (Bc, d)

        # computing block partial score
        # equivalent to Q_1 @ K_j^T GEMM
        # if q_len == 1, then one should call 'single-query optimized kernel' istead of this one.
        S_i_j = tl.dot(Q_i, tl.trans(K_j), out_dtype=tl.float32) * d_inv  # (Br, d)

        # overflowed i, j-index -inf masking
        neg_inf = tl.full((Br, Bc), -float("inf"), dtype=tl.float32)  # (Br, Bc)
        S_i_j = tl.where(j * Bc + tl.arange(0, Bc) < kv_len, S_i_j, neg_inf)  # (Br, Bc)
        S_i_j = tl.where(l_offs < q_len, S_i_j, neg_inf)  # (Br, Bc)

        if is_causal:
            S_i_j = tl.where(l_offs >= j * Bc + tl.arange(0, Bc), S_i_j, neg_inf)  # condition i >= j

        # computing partial-scaled softmax
        S_i_j_rowmax = tl.max(S_i_j, axis=-1)  # (Br,)
        m_i_j_new = tl.maximum(m_i_j, S_i_j_rowmax)  # (Br,)
        P_i_j = tl.exp(S_i_j - m_i_j_new[:, None])  # (Br, Bc)

        # renormalized score-value mm
        scale = tl.exp(m_i_j - m_i_j_new)  # (Br,)
        l_i_j = scale * l_i_j + tl.sum(P_i_j, axis=-1)  # (Br,)
        O_i_j = scale[:, None] * O_i_j + tl.dot(  # (Br, Bc) @ (Bc, d) -> (Br, d)
            P_i_j, tl.cast(V_j, dtype=tl.float32), out_dtype=tl.float32
        )
        m_i_j = m_i_j_new

    # writing accumulated output to DRAM
    out = (1 / l_i_j)[:, None] * O_i_j  # (Br, d)
    out_ptrs = O_ptr + b * Q_st0 + h * Q_st1 + l_offs * Q_st2 + d_offs * Q_st3
    tl.store(out_ptrs, out, mask=l_offs < q_len)


def get_cuda_autotune_config_2():
    Bcs = [256, 128, 64, 32, 16]
    num_stages = [5, 4, 3]
    num_warps = [8, 4, 2]
    return list(
        triton.Config(kwargs={"Bc": Bc}, num_stages=ns, num_warps=nw)
        for Bc, ns, nw in itertools.product(Bcs, num_stages, num_warps)
    )


# @triton.autotune(
#     configs=get_cuda_autotune_config_2(),
#     key=["kv_len", "d"],
# )
@triton.jit
def my_flash_attn_kernel_single_query(  # TODO: solve when param::d is not a power of 2.
    Q_ptr, K_ptr, V_ptr, O_ptr,
    kv_len,
    Q_st0, Q_st1, Q_st2, Q_st3,
    K_st0, K_st1, K_st2, K_st3,
    V_st0, V_st1, V_st2, V_st3,
    d: tl.constexpr,
    num_kv_groups: tl.constexpr,
    Bc: tl.constexpr,
):
    # Parallel grid for each batch, head
    # Parallel grid for each i * Br: (i+1) * Br segment in sequence length dimension
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)

    tl.assume(h >= 0)
    tl.assume(Q_st0 >= 0)
    tl.assume(Q_st1 >= 0)
    tl.assume(Q_st2 >= 0)
    tl.assume(Q_st3 >= 0)
    tl.assume(K_st0 >= 0)
    tl.assume(K_st1 >= 0)
    tl.assume(K_st2 >= 0)
    tl.assume(K_st3 >= 0)
    tl.assume(V_st0 >= 0)
    tl.assume(V_st1 >= 0)
    tl.assume(V_st2 >= 0)
    tl.assume(V_st3 >= 0)
    tl.assume(num_kv_groups >= 1)

    # loading Q_i panel to SRAM
    # same with ~[b, h, 0, :]
    # only using 1-dimensional broadcasting will load 4D-tensor as 1D.
    d_offs = tl.arange(0, d)
    Q_i_ptr = Q_ptr + b * Q_st0 + h * Q_st1 + d_offs * Q_st3

    Q_i = tl.load(Q_i_ptr)  # (d,)

    # initializing accumulator tensors in SRAM
    O_i_j = tl.zeros((d,), dtype=tl.float32)  # (d,)
    l_i_j = tl.zeros((), dtype=tl.float32)  # ()
    m_i_j = tl.full((), -1e12, dtype=tl.float32)  # ()

    Tc = tl.cdiv(kv_len, Bc)
    d_inv = 1 / (d ** 0.5)

    # block-tiled online-softmax & matmul V-tile
    for j in range(Tc):
        # loading K_j, V_j to SRAM
        # same with ~[b, h, j * Bc: j * Bc + Bc, :]
        block_l_offs =  j * Bc + tl.arange(0, Bc)[:, None]
        block_d_offs = tl.arange(0, d)[None, :]
        K_j_ptr = K_ptr + b * K_st0 + (h // num_kv_groups) * K_st1 + \
            block_l_offs * K_st2 + block_d_offs * K_st3
        V_j_ptr = V_ptr + b * V_st0 + (h // num_kv_groups) * V_st1 + \
            block_l_offs * V_st2 + block_d_offs * V_st3

        K_j = tl.load(K_j_ptr, mask=block_l_offs < kv_len, other=0.0)  # (Bc, d)
        V_j = tl.load(V_j_ptr, mask=block_l_offs < kv_len, other=0.0)  # (Bc, d)

        # computing block partial score
        # below is equivalent to K_j @ q_i GEMV.
        S_i_j = tl.sum(Q_i[None, :] * K_j, axis=-1) * d_inv  # (Bc,)

        # overflowed j-index -inf masking
        neg_inf = tl.full((Bc,), -float("inf"), dtype=tl.float32)  # (Bc,)
        S_i_j = tl.where(j * Bc + tl.arange(0, Bc) < kv_len, S_i_j, neg_inf)  # (Bc,)

        # computing partial-scaled softmax
        S_i_j_rowmax = tl.max(S_i_j, axis=-1)  # ()
        m_i_j_new = tl.maximum(m_i_j, S_i_j_rowmax)  # ()
        P_i_j = tl.exp(S_i_j - m_i_j_new)  # (Bc,)

        # renormalized score-value mm
        scale = tl.exp(m_i_j - m_i_j_new)  # ()
        l_i_j = scale * l_i_j + tl.sum(P_i_j, axis=-1)  # (Bc,)
        pv = P_i_j[:, None] * tl.cast(V_j, dtype=tl.float32)  # (Bc, 1) * (Bc, d) -> (Bc, d)
        pv_sum = tl.sum(pv, axis=0)  # (d,)
        O_i_j = scale * O_i_j + pv_sum  # (d,)

        # # FP32 accumulation
        # Vj = tl.cast(V_j, tl.float32)
        # acc = scale * O_i_j
        # for r in range(Bc):
        #     w = P_i_j[r]
        #     acc += Vj[r, :] * w
        # O_i_j = acc

        m_i_j = m_i_j_new

    # writing accumulated output to DRAM
    out = (1 / l_i_j) * O_i_j
    out_ptrs = O_ptr + b * Q_st0 + h * Q_st1 + d_offs * Q_st3
    tl.store(out_ptrs, out)


def my_flash_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, H = Q.shape[0], Q.shape[1]
    q_len, kv_len = Q.shape[2], K.shape[2]
    d = Q.shape[-1]
    num_kv_groups = int(H // K.shape[1])

    O = torch.empty_like(Q)

    if q_len == 1:
        Bc = 256
        grid = lambda meta: (B, H,)
        my_flash_attn_kernel_single_query[grid](  # 2d-grid launching
            Q, K, V, O,
            kv_len,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            d=d,
            Bc=Bc,
            num_kv_groups=num_kv_groups,
        )
    else:
        Br = 16
        Bc = 64
        grid = lambda meta: (B, H, triton.cdiv(q_len, Br), )
        my_flash_attn_kernel[grid](  # 3d-grid launching
            Q, K, V, O,
            q_len, kv_len,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            d=d,
            is_causal=is_causal,
            num_kv_groups=num_kv_groups,
            Br=Br,
            Bc=Bc,
            num_warps=8,
            num_stages=1,
        )
    return O, None


# TODO: triton ptx kernel integration is now under dev
# add_()
mm_()
