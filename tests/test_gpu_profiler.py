import time
from unittest.mock import MagicMock, patch

import pytest

from src.gpu_profiler import GpuProfiler, ProfileResult, _NvmlMemoryInfo


class TestProfileResult:
    """Verify ProfileResult dataclass."""

    def test_fields(self) -> None:
        r = ProfileResult(
            baseline_bytes=1000,
            peak_bytes=2000,
            delta_bytes=1000,
            total_bytes=10000,
            num_gpus=0.15,
            samples=50,
            poll_interval_ms=5.0,
        )
        assert r.baseline_bytes == 1000
        assert r.peak_bytes == 2000
        assert r.delta_bytes == 1000
        assert r.total_bytes == 10000
        assert r.num_gpus == 0.15
        assert r.samples == 50

    def test_frozen(self) -> None:
        r = ProfileResult(
            baseline_bytes=0,
            peak_bytes=0,
            delta_bytes=0,
            total_bytes=0,
            num_gpus=0.0,
            samples=0,
            poll_interval_ms=5.0,
        )
        with pytest.raises(AttributeError):
            r.num_gpus = 0.5  # type: ignore[misc]


class TestGpuProfiler:
    """Test GpuProfiler with mocked NVML calls."""

    def _make_profiler(
        self, total_bytes: int = 24 * 1024**3
    ) -> tuple[GpuProfiler, MagicMock]:
        """Create a GpuProfiler with mocked NVML.

        Parameters
        ----------
        total_bytes : int
            Simulated total GPU memory.

        Returns
        -------
        tuple[GpuProfiler, MagicMock]
            ``(profiler, mock_nvml)`` for controlling memory responses.
        """
        mock_nvml = MagicMock()
        mock_nvml.nvmlInit_v2.return_value = 0
        mock_nvml.nvmlDeviceGetHandleByIndex_v2.return_value = 0
        mock_nvml.nvmlShutdown.return_value = 0

        # Track call count for varying responses
        self._mem_call_count = 0
        self._mem_values: list[int] = [1 * 1024**3]  # Default: 1 GiB

        def fake_get_memory(handle, mem_ptr):  # noqa: ANN001
            mem = _NvmlMemoryInfo.from_address(
                id(mem_ptr) if not hasattr(mem_ptr, "_obj") else id(mem_ptr._obj)
            )
            # We can't easily write to the ctypes struct through the pointer
            # in a mock, so we use a different approach
            return 0

        # Set up memory info mock to return controlled values
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = 0

        with patch("src.gpu_profiler.ctypes") as mock_ctypes:
            # Make find_library return a name
            mock_ctypes.util.find_library.return_value = "libnvidia-ml.so.1"
            mock_ctypes.CDLL.return_value = mock_nvml
            mock_ctypes.c_void_p.return_value = MagicMock()
            mock_ctypes.c_ulonglong = type(MagicMock())

            # Create a fake memory struct
            fake_mem = MagicMock()
            fake_mem.total = total_bytes
            fake_mem.free = total_bytes - 1 * 1024**3
            fake_mem.used = 1 * 1024**3

            # Bypass __init__ and set up manually
            profiler = GpuProfiler.__new__(GpuProfiler)
            profiler._device_index = 0
            profiler._poll_interval_ms = 5.0
            profiler._margin = 1.15
            profiler._nvml = mock_nvml
            profiler._handle = MagicMock()
            profiler._total_bytes = total_bytes

            import threading

            profiler._lock = threading.Lock()
            profiler._polling = False
            profiler._baseline = 0
            profiler._peak_used = 0
            profiler._sample_count = 0
            profiler._thread = None

        return profiler, mock_nvml

    def test_start_returns_baseline(self) -> None:
        profiler, mock_nvml = self._make_profiler()

        # Mock _get_memory to return known values
        baseline_mem = MagicMock()
        baseline_mem.total = 24 * 1024**3
        baseline_mem.free = 20 * 1024**3
        baseline_mem.used = 4 * 1024**3
        profiler._get_memory = MagicMock(return_value=baseline_mem)

        baseline = profiler.start()
        assert baseline == 4 * 1024**3

        # Stop immediately
        result = profiler.stop()
        assert result.baseline_bytes == 4 * 1024**3

    def test_stop_returns_profile_result(self) -> None:
        profiler, _ = self._make_profiler(total_bytes=24 * 1024**3)

        base_mem = MagicMock()
        base_mem.used = 2 * 1024**3
        base_mem.total = 24 * 1024**3
        base_mem.free = 22 * 1024**3
        profiler._get_memory = MagicMock(return_value=base_mem)

        profiler.start()
        # Simulate peak by directly setting it
        with profiler._lock:
            profiler._peak_used = 5 * 1024**3

        result = profiler.stop()
        assert isinstance(result, ProfileResult)
        assert result.peak_bytes == 5 * 1024**3
        assert result.delta_bytes == 3 * 1024**3  # 5 - 2
        assert result.total_bytes == 24 * 1024**3
        assert result.num_gpus > 0

    def test_num_gpus_calculation(self) -> None:
        profiler, _ = self._make_profiler(total_bytes=24 * 1024**3)

        base_mem = MagicMock()
        base_mem.used = 0
        base_mem.total = 24 * 1024**3
        base_mem.free = 24 * 1024**3
        profiler._get_memory = MagicMock(return_value=base_mem)

        profiler.start()
        with profiler._lock:
            # Simulate 6 GiB peak = 25% of 24 GiB
            profiler._peak_used = 6 * 1024**3

        result = profiler.stop()
        # 6/24 = 0.25, * 1.15 margin = 0.2875, rounded to 0.30
        assert result.num_gpus == 0.30

    def test_num_gpus_capped_at_1(self) -> None:
        profiler, _ = self._make_profiler(total_bytes=8 * 1024**3)

        base_mem = MagicMock()
        base_mem.used = 0
        base_mem.total = 8 * 1024**3
        base_mem.free = 8 * 1024**3
        profiler._get_memory = MagicMock(return_value=base_mem)

        profiler.start()
        with profiler._lock:
            # 10 GiB on 8 GiB GPU — exceeds total
            profiler._peak_used = 10 * 1024**3

        result = profiler.stop()
        assert result.num_gpus <= 1.0

    def test_shutdown(self) -> None:
        profiler, mock_nvml = self._make_profiler()
        profiler.shutdown()
        mock_nvml.nvmlShutdown.assert_called_once()

    def test_used_bytes_property(self) -> None:
        profiler, _ = self._make_profiler()
        mem = MagicMock()
        mem.used = 3 * 1024**3
        profiler._get_memory = MagicMock(return_value=mem)
        assert profiler.used_bytes == 3 * 1024**3

    def test_total_bytes_property(self) -> None:
        profiler, _ = self._make_profiler(total_bytes=16 * 1024**3)
        assert profiler.total_bytes == 16 * 1024**3
