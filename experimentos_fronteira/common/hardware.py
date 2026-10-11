"""
Guards e helpers para uso responsável do M4 Pro.

Uso típico:
    from common.hardware import device, limit_mps_memory, cleanup_mps
    limit_mps_memory(fraction=0.6)
    # ... treino ...
    cleanup_mps()
"""
from __future__ import annotations

import os
import warnings


def check_mps() -> bool:
    """Retorna True se MPS está disponível e ativo."""
    try:
        import torch
        return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())
    except ImportError:
        return False


def device():
    """Retorna torch.device otimizado para Apple Silicon (mps > cuda > cpu)."""
    import torch
    if check_mps():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def limit_mps_memory(fraction: float = 0.6) -> None:
    """Limita PyTorch a usar no máximo `fraction` da GPU Metal. Ignora se sem MPS."""
    try:
        import torch
        if check_mps() and hasattr(torch.mps, "set_per_process_memory_fraction"):
            torch.mps.set_per_process_memory_fraction(fraction)
            print(f"[hardware] MPS memory fraction = {fraction}")
    except Exception as e:
        warnings.warn(f"Não foi possível limitar memória MPS: {e}")


def cleanup_mps() -> None:
    """Libera cache MPS entre experimentos."""
    try:
        import torch
        if check_mps():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    except Exception:
        pass


def set_omp_threads(n: int = 4) -> None:
    """Limita threads OpenMP para não saturar CPU."""
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)


def print_hardware_summary() -> None:
    """Resumo do hardware atual (útil no início de cada experimento)."""
    import platform
    print(f"[hardware] OS: {platform.system()} {platform.release()}")
    print(f"[hardware] Python: {platform.python_version()}")
    try:
        import torch
        print(f"[hardware] PyTorch: {torch.__version__}")
        print(f"[hardware] MPS disponível: {check_mps()}")
        print(f"[hardware] Device selecionado: {device()}")
    except ImportError:
        print("[hardware] PyTorch não instalado")
    try:
        import mlx
        mlx_ver = getattr(mlx, "__version__", None) or getattr(mlx, "version", "instalado (versão desconhecida)")
        print(f"[hardware] MLX: {mlx_ver}")
    except ImportError:
        print("[hardware] MLX não instalado (opcional)")


def memory_pressure_ok() -> bool:
    """
    Heurística simples de pressão de memória.
    Retorna False se deve-se reduzir batch size.
    """
    try:
        import subprocess
        out = subprocess.check_output(
            ["vm_stat"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if "Pages free" in line:
                pages = int(line.split(":")[1].strip().rstrip("."))
                # cada page = 16 KB em Apple Silicon
                free_gb = pages * 16 * 1024 / 1e9
                return free_gb > 2.0  # pelo menos 2 GB livres
    except Exception:
        pass
    return True


if __name__ == "__main__":
    print_hardware_summary()
    print(f"[hardware] Pressão de memória OK: {memory_pressure_ok()}")
