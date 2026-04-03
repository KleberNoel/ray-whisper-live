import argparse
import logging
import time

import ray
from ray import serve

from transcriber_deployment import WhisperTranscriber
from server_deployment import WhisperLiveServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ray Serve WhisperLive")
    parser.add_argument(
        "--model-size", default="large-v3-turbo", help="Whisper model size"
    )
    args = parser.parse_args()

    ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True)
    logger.info(
        "Ray: %.0f GPUs, %.0f CPUs",
        ray.available_resources().get("GPU", 0),
        ray.available_resources().get("CPU", 0),
    )

    transcriber = WhisperTranscriber.bind(model_size=args.model_size)
    server = WhisperLiveServer.bind(transcriber_handle=transcriber)
    serve.run(server, route_prefix="/")

    logger.info(
        "WhisperLive ready: ws://localhost:8000/listen (model=%s)", args.model_size
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
