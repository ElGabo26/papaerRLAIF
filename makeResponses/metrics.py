import os
import psutil


try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


def medir_recursos(pid=None):
    """
    Devuelve el consumo de RAM y GPU de un proceso.

    Parámetros
    ----------
    pid : int, opcional
        PID del proceso. Si es None utiliza el proceso actual.

    Retorna
    -------
    dict
    """

    if pid is None:
        pid = os.getpid()

    proceso = psutil.Process(pid)

    # RAM
    memoria = proceso.memory_info()

    ram_bytes = memoria.rss
    ram_mb = ram_bytes / 1024**2
    ram_gb = ram_bytes / 1024**3

    gpu_mb = 0
    gpu_gb = 0
    gpu_id = None

    if NVML_AVAILABLE:

        n_gpus = pynvml.nvmlDeviceGetCount()

        for i in range(n_gpus):

            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            try:
                procesos = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except:
                procesos = []

            for p in procesos:

                if p.pid == pid:

                    gpu_id = i
                    gpu_mb = p.usedGpuMemory / 1024**2
                    gpu_gb = p.usedGpuMemory / 1024**3

                    break

    return {
        "pid": pid,
        "ram_mb": round(ram_mb, 2),
        "ram_gb": round(ram_gb, 3),
        "gpu_id": gpu_id,
        "gpu_mb": round(gpu_mb, 2),
        "gpu_gb": round(gpu_gb, 3),
    }