import ctypes
import ctypes.util
import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class _NvmlMemoryInfo(ctypes.Structure):
    """Mirror of ``nvmlMemory_t`` from NVML."""

    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


@dataclass(frozen=True)
class ProfileResult:
    """Peak GPU memory recorded during a profiling window.

    Parameters
    ----------
    baseline_bytes : int
        GPU memory used before the profiled operation.
    peak_bytes : int
        Highest GPU memory observed during the operation.
    delta_bytes : int
        ``peak_bytes - baseline_bytes`` — the operation's footprint.
    total_bytes : int
        Total GPU memory on the device.
    num_gpus : float
        Recommended Ray ``num_gpus`` fraction (delta / total, with margin).
    samples : int
        Number of memory samples collected.
    poll_interval_ms : float
        Polling interval used during profiling.
    """

    baseline_bytes: int
    peak_bytes: int
    delta_bytes: int
    total_bytes: int
    num_gpus: float
    samples: int
    poll_interval_ms: float


class GpuProfiler:
    """Poll NVML GPU memory at millisecond intervals to capture peak usage.

    Uses ``ctypes`` to call NVML directly — no ``pynvml`` dependency.

    Parameters
    ----------
    device_index : int
        GPU device ordinal (default ``0``).
    poll_interval_ms : float
        Polling interval in milliseconds (default ``5.0``).
    margin : float
        Safety margin multiplier for ``num_gpus`` (default ``1.15``,
        i.e. 15% headroom).
    """

    def __init__(
        self,
        device_index: int = 0,
        poll_interval_ms: float = 5.0,
        margin: float = 1.15,
    ) -> None:
        self._device_index = device_index
        self._poll_interval_ms = poll_interval_ms
        self._margin = margin

        # Load NVML
        lib_name = ctypes.util.find_library("nvidia-ml")
        if lib_name is None:
            raise RuntimeError("NVML library not found (libnvidia-ml.so)")
        self._nvml = ctypes.CDLL(lib_name)

        # Initialize NVML and get device handle
        self._check(self._nvml.nvmlInit_v2(), "nvmlInit_v2")
        self._handle = ctypes.c_void_p()
        self._check(
            self._nvml.nvmlDeviceGetHandleByIndex_v2(
                device_index, ctypes.byref(self._handle)
            ),
            "nvmlDeviceGetHandleByIndex_v2",
        )

        # Cache total memory
        mem = self._get_memory()
        self._total_bytes = mem.total
        logger.info(
            "GpuProfiler: device %d, total=%.2f GiB, poll=%.1f ms",
            device_index,
            self._total_bytes / (1024**3),
            poll_interval_ms,
        )

        # Polling state
        self._lock = threading.Lock()
        self._polling = False
        self._baseline: int = 0
        self._peak_used: int = 0
        self._sample_count: int = 0
        self._thread: threading.Thread | None = None

    def _check(self, ret: int, fn_name: str) -> None:
        """Raise on non-zero NVML return code.

        Parameters
        ----------
        ret : int
            NVML return code.
        fn_name : str
            Function name for error message.
        """
        if ret != 0:
            raise RuntimeError(f"NVML {fn_name} failed with code {ret}")

    def _get_memory(self) -> _NvmlMemoryInfo:
        """Query current GPU memory usage.

        Returns
        -------
        _NvmlMemoryInfo
            NVML memory struct with ``total``, ``free``, ``used``.
        """
        mem = _NvmlMemoryInfo()
        self._check(
            self._nvml.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(mem)),
            "nvmlDeviceGetMemoryInfo",
        )
        return mem

    @property
    def used_bytes(self) -> int:
        """Current GPU memory used in bytes."""
        return self._get_memory().used

    @property
    def total_bytes(self) -> int:
        """Total GPU memory in bytes."""
        return self._total_bytes

    def _poll_loop(self) -> None:
        """Background thread: poll memory and track peak."""
        interval = self._poll_interval_ms / 1000.0
        while True:
            with self._lock:
                if not self._polling:
                    break
            used = self._get_memory().used
            with self._lock:
                if used > self._peak_used:
                    self._peak_used = used
                self._sample_count += 1
            time.sleep(interval)

    def start(self) -> int:
        """Begin polling. Returns the baseline memory in bytes.

        Returns
        -------
        int
            Current GPU memory usage (baseline) when polling starts.
        """
        baseline = self._get_memory().used
        with self._lock:
            self._polling = True
            self._baseline = baseline
            self._peak_used = baseline
            self._sample_count = 0
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return baseline

    def stop(self) -> ProfileResult:
        """Stop polling and return the profiling result.

        Returns
        -------
        ProfileResult
            Contains baseline, peak, delta, and recommended ``num_gpus``.
        """
        with self._lock:
            self._polling = False
            peak = self._peak_used
            baseline = self._baseline
            samples = self._sample_count

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        delta = max(0, peak - baseline)

        # Calculate num_gpus fraction with safety margin
        raw_fraction = delta / self._total_bytes if self._total_bytes > 0 else 0.0
        margined = raw_fraction * self._margin
        # Round up to nearest 0.05 for clean Ray allocation
        num_gpus = min(1.0, round(margined * 20 + 0.5) / 20)

        result = ProfileResult(
            baseline_bytes=baseline,
            peak_bytes=peak,
            delta_bytes=delta,
            total_bytes=self._total_bytes,
            num_gpus=num_gpus,
            samples=samples,
            poll_interval_ms=self._poll_interval_ms,
        )

        logger.info(
            "Profile: baseline=%.2f GiB, peak=%.2f GiB, delta=%.2f GiB, "
            "num_gpus=%.2f (%d samples @ %.1f ms)",
            result.baseline_bytes / (1024**3),
            result.peak_bytes / (1024**3),
            result.delta_bytes / (1024**3),
            result.num_gpus,
            result.samples,
            result.poll_interval_ms,
        )
        return result

    def shutdown(self) -> None:
        """Release NVML resources."""
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass
