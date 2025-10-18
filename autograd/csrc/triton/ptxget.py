import triton

DESC = "[PTXGET]"


class CompiledKernelExtractor:
    reg = []

    @classmethod
    def extract(cls, *args, grid, **kwargs):
        """ Runs triton jitted kernels to compile & extract ptx IR """

        def decorator(kernel: triton.JITFunction):
            print(f"{DESC} Compiling ptx kernel '{kernel.__name__}'")
            kernel[grid](*args, **kwargs)
            for signatures, compiled_kernel in kernel.device_caches[0][0].items():
                sigs = eval(signatures.split("{")[0])
                non_const_sig_idx = []
                for i, (typ, _) in enumerate(sigs):
                    if typ != "constexpr":
                        non_const_sig_idx.append(i)
                cls.reg.append(
                    (kernel.__name__, non_const_sig_idx, compiled_kernel.asm)
                )
                break
            return kernel
        return decorator


if __name__ == "__main__":
    import os
    import subprocess

    import triton_kernels

    DUMP_DIR = "autograd/csrc/triton/bin"
    os.makedirs(DUMP_DIR, exist_ok=True)

    def ptx_to_obj(ptx_path: str, obj_path: str, sym_prefix: str):
        syms = [
            "_binary_" + ptx_path.replace("/", "_").replace(".", "_") + "_start",
            "_binary_" + ptx_path.replace("/", "_").replace(".", "_") + "_end",
            "_binary_" + ptx_path.replace("/", "_").replace(".", "_") + "_size"
        ]
        args = [
            "objcopy",
            "-I", "binary",
            "-O", "elf64-x86-64",
            "--rename-section", ".data=.rodata,alloc,load,readonly,data,contents",
            ptx_path, obj_path
        ]
        for s in syms:
            new = sym_prefix + "_" + s.split("_")[-1]
            args.extend(["--redefine-sym", f"{s}={new}"])
        subprocess.check_call(args)

    for name, non_const_sig_idx, asm in triton_kernels.ext.reg:
        dir_path = f"{DUMP_DIR}/{name}"
        os.makedirs(dir_path, exist_ok=True)
        
        with open(f"{dir_path}/sigs.txt", "w") as f:
            f.write(str(non_const_sig_idx))
        
        for k, v in asm.items():
            desc = "wb" if k == "cubin" else "w"
            fname = f"{dir_path}/{name}.{k}"
            with open(fname, desc) as f:
                f.write(v)
            if k == "ptx":
                print(DESC, f"Converting '{name}.ptx' -> '{name}.ptx.o'")
                ptx_to_obj(fname, fname + ".o", sym_prefix="_ptx_" + name)
                print(DESC, f"'{fname}' symbols: ")
                os.system(f"nm {fname}.o")  # inspect object symbols
