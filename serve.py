"""Ray Serve WhisperLive - Entry point for serving."""

import argparse
import logging

import ray
from ray import serve

from vad_deployment import VADDetector
from transcriber_deployment import WhisperTranscriberDeployment
from server_deployment import WhisperLiveServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_app(
    vad_threshold: float = 0.5,
    model_size: str = "large-v3-turbo",
    beam_size: int = 5,
    no_speech_threshold: float = 0.45,
):
    """
    Build the Ray Serve application graph.

    Args:
        vad_threshold: VAD speech probability threshold
        model_size: Whisper model size
        beam_size: Beam size for decoding
        no_speech_threshold: Threshold for filtering silent segments

    Returns:
        Ray Serve deployment handle
    """
    logger.info("Building WhisperLive application...")

    vad = VADDetector.options(name="VADDetector").bind(threshold=vad_threshold)

    transcriber = WhisperTranscriberDeployment.options(name="WhisperTranscriber").bind(
        model_size=model_size,
        beam_size=beam_size,
        no_speech_threshold=no_speech_threshold,
    )

    server = WhisperLiveServer.options(name="WhisperLiveServer").bind(
        vad_handle=vad,
        transcriber_handle=transcriber,
    )

    server_handle = serve.run(server, route_prefix="/")

    logger.info("WhisperLive application deployed")
    logger.info(f"  VAD threshold: {vad_threshold}")
    logger.info(f"  Model: {model_size}")
    logger.info(f"  Beam size: {beam_size}")
    logger.info(f"  No-speech threshold: {no_speech_threshold}")
    logger.info(f"  WebSocket endpoint: ws://<host>:8000/listen")

    return server_handle


def main():
    parser = argparse.ArgumentParser(description="Ray Serve WhisperLive")
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD speech probability threshold",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="large-v3-turbo",
        help="Whisper model size",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.45,
        help="Threshold for filtering silent segments",
    )
    args = parser.parse_args()

    logger.info("Initializing Ray...")
    ray.init(
        num_cpus=4,
        num_gpus=1,
        ignore_reinit_error=True,
    )

    logger.info("Ray initialized")
    logger.info(f"  GPUs: {ray.available_resources().get('GPU', 0)}")
    logger.info(f"  CPUs: {ray.available_resources().get('CPU', 0)}")

    build_app(
        vad_threshold=args.vad_threshold,
        model_size=args.model_size,
        beam_size=args.beam_size,
        no_speech_threshold=args.no_speech_threshold,
    )

    logger.info("Server running. Press Ctrl+C to stop.")
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
