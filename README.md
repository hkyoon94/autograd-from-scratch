### `autograd`: “Lightweight Autograd Engine with Visualized Computation Graph and Automatic Backpropagation”

#### 🔹 Overview
A minimal yet expressive autograd engine built entirely in **C++** with Python bindings,  
featuring PyTorch-style APIs, graph visualization, and efficient automatic back-propagation.

#### 🔹 Core Features (For details, see: <Demo>)
- ![alt text](autograd_overview.png)
- Lightweight, **DAG-based Computation Graph** tracking for versatile tensor operations, **including some used forward / backward operators** such as Softmax-Crossentropy reduction.
- **Shape-aware Computation Graph visualization** via `Graphviz`
  - (including explicit visualization Parameter, and Non-parameter leaf tensors.)
- **PyTorch-style API**: `Tensor`, `Optimizer`, `autograd.backward()`, also supports `no_grad()` context.
- **Efficient Topological Backward Traversal** implemented in **C++ backend**
- **Benchmark:** ~**1.2–1.3× faster than PyTorch for small-model workloads** (See section. #2)


#### 🔹 Under Development
1. Low-latency dispatch table in C++ backend
2. Enhanced `Function` API bindings between Python <--> C++ backend
3. Kernels for Additional operations such as `split`(chunk), `Conv`, `bmm` and `einsum` (with backward).
4. `Triton`-based PTX compilation for `CUDA` backend
5. Computation graph analyzer for operation fusion & model compilation
6. More versatile fused operators

---

### 🧩 Demo Contents
1. **Computation Graph Visualization**  
   - Shape-aware visualized DAG for arbitrary models showing relationships of `Tensor` and `Function` nodes.
2. **Training Benchmark**  
   - CPU backend benchmark vs. `PyTorch` showing correctness (accuracy ~0.96) and lower runtime overhead.

    Note: This benchmark uses small tensor workloads (≤64×64), where framework-level overhead becomes the dominant factor.\
    The result demonstrates the lightweight design efficiency of the custom C++ autograd engine:
    #### 📈 Benchmark Results Summary
    | Model | #Params | Accuracy | Forward total | Backward total | Speedup |
    |--------|----------|-----------|-----------|------------|------------|
    | PyTorch (Single-thread) | 4,736 | 0.95-0.96 | 3.82s | 10.41s | – |
    | `autograd` (C++) | 4,736 | 0.95-0.96 | **3.31s** | **8.00s** | **1.2-1.3x** |
    
    Note: Under PyTorch’s default multi-threaded mode, \
    total runtime increased to **12.07s (forward)** and **32.49s (backward)** due to threading overhead on small tensor workloads.

---

### 🧠 Key Takeaway
> A **lightweight autograd engine** demonstrating correct graph-based backpropagation,  
> reduced runtime overhead, and extendability toward compiler-level optimizations.