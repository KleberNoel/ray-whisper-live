from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.config import SAMPLE_RATE


@pytest.fixture
def mock_websocket() -> AsyncMock:
    """Create a mock FastAPI WebSocket."""
    ws = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_bytes = AsyncMock()
    ws.receive_json = AsyncMock()
    ws.close = AsyncMock()
    ws.client_state = MagicMock()
    ws.client_state.name = "CONNECTED"
    return ws


@pytest.fixture
def silence_1s() -> np.ndarray:
    """1 second of silence at 16 kHz."""
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


@pytest.fixture
def tone_1s() -> np.ndarray:
    """1 second of 440 Hz sine tone at 16 kHz."""
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)
