"""Measure the download rate of one GitHub runner from the OSM planet mirrors.

Run by the workflow `runner-network.yml`, once per job. Prints one JSON line about the
runner, then one JSON line per measurement. Each measurement reads 1 GiB of the planet file
by HTTP byte ranges and discards the bytes, as the scgraph pipeline reads the planet without
storing it.

Usage: python -I measure.py JOB ATTEMPT
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

# Bytes per measurement; PROBE_BYTES only shrinks it for a dry run of the script
GIB = int(os.environ.get("PROBE_BYTES", 1 << 30))
PATH = "planet/pbf/2026/planet-261102.osm.pbf"
MIRRORS = {
    # The mirror that scgraph's planet source names (stage1_extract/osm.py, dev)
    "eu": f"https://osm-planet-eu-central-1.s3.dualstack.eu-central-1.amazonaws.com/{PATH}",
    "us": f"https://osm-planet-us-west-2.s3.dualstack.us-west-2.amazonaws.com/{PATH}",
}
SIZE = 95715394709
ETAG = '"710883a12aa9b9461d04dabf389536b5-5706"'
# (mirror, client, parallel streams)
CONFIGS = [
    ("eu", "curl", 1),
    ("eu", "curl", 4),
    ("us", "curl", 1),
    ("us", "curl", 4),
    ("eu", "requests", 1),
]
TIMEOUT_S = 600  # per measurement: 1 GiB at 1.8 MB/s


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_curl(url: str, start: int, length: int) -> dict:
    """One stream with curl: bytes received, status, ETag and Content-Range."""
    with tempfile.NamedTemporaryFile("r") as headers:
        done = subprocess.run(
            [
                "curl", "-sS", "-A", "scgraph-runner-probe",
                "-r", f"{start}-{start + length - 1}",
                "--max-time", str(TIMEOUT_S),
                "-D", headers.name, "-o", "/dev/null",
                "-w", "%{http_code} %{size_download}",
                url,
            ],
            capture_output=True,
            text=True,
        )
        lines = headers.read().splitlines()
    status, received = (done.stdout.split() + ["0", "0"])[:2]
    found = {}
    for line in lines:
        name, _, value = line.partition(":")
        found[name.strip().lower()] = value.strip()
    return {
        "exit": done.returncode,
        "status": int(status),
        "bytes": int(float(received)),
        "etag": found.get("etag"),
        "content_range": found.get("content-range"),
        "error": done.stderr.strip()[-300:] or None,
    }


def read_requests(url: str, start: int, length: int) -> dict:
    """One stream with requests, read in 1 MiB pieces as a pipe consumer would."""
    received, result = 0, {"exit": 0, "error": None}
    began = time.perf_counter()
    try:
        with requests.get(
            url,
            headers={
                "User-Agent": "scgraph-runner-probe",
                "Range": f"bytes={start}-{start + length - 1}",
            },
            stream=True,
            timeout=(10.0, 60.0),
        ) as answer:
            result |= {
                "status": answer.status_code,
                "etag": answer.headers.get("ETag"),
                "content_range": answer.headers.get("Content-Range"),
            }
            for piece in answer.iter_content(1 << 20):
                received += len(piece)
                if time.perf_counter() - began > TIMEOUT_S:
                    raise TimeoutError("measurement over its time limit")
    except (requests.RequestException, OSError, TimeoutError) as error:
        result |= {"exit": 1, "error": repr(error)[-300:]}
    return {"status": 0, "etag": None, "content_range": None} | result | {
        "bytes": received
    }


def measure(mirror: str, client: str, streams: int, start: int) -> dict:
    """Read 1 GiB from `start`, split in `streams` contiguous parts read at once."""
    url = MIRRORS[mirror]
    part = GIB // streams
    read = read_curl if client == "curl" else read_requests
    began = time.perf_counter()
    with ThreadPoolExecutor(streams) as pool:
        parts = list(
            pool.map(lambda k: read(url, start + k * part, part), range(streams))
        )
    seconds = time.perf_counter() - began
    received = sum(p["bytes"] for p in parts)
    expected = [f"bytes {start + k * part}-{start + (k + 1) * part - 1}/{SIZE}"
                for k in range(streams)]
    # Conservation: every byte asked was received, from the same file
    intact = all(
        p["exit"] == 0 and p["status"] == 206 and p["bytes"] == part
        and p["etag"] == ETAG and p["content_range"] == want
        for p, want in zip(parts, expected)
    )
    return {
        "mirror": mirror,
        "client": client,
        "streams": streams,
        "start": start,
        "bytes": received,
        "seconds": round(seconds, 3),
        "mb_per_s": round(received / seconds / 1e6, 2),
        "intact": intact,
        "errors": [p["error"] for p in parts if p["error"]],
        "finished": now(),
    }


def describe_runner(job: int, attempt: int) -> dict:
    with open("/proc/meminfo") as file:
        memory_kb = int(file.readline().split()[1])
    disks = {}
    for mount in ("/", "/mnt", os.environ.get("RUNNER_TEMP", "/tmp")):
        if os.path.exists(mount):
            usage = shutil.disk_usage(mount)
            disks[mount] = {"total": usage.total, "free": usage.free}
    curl = subprocess.run(["curl", "--version"], capture_output=True, text=True)
    return {
        "kind": "runner",
        "started": now(),
        "job": job,
        "attempt": attempt,
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "commit": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "image": os.environ.get("ImageOS"),
        "image_version": os.environ.get("ImageVersion"),
        "cpus": os.cpu_count(),
        "memory_bytes": memory_kb * 1024,
        "disks": disks,
        "python": platform.python_version(),
        "requests": requests.__version__,
        "curl": curl.stdout.split("\n")[0],
    }


def main() -> None:
    job, attempt = int(sys.argv[1]), int(sys.argv[2])
    print(json.dumps(describe_runner(job, attempt)), flush=True)
    # The order of the configurations rotates between jobs and attempts
    shift = (job + 3 * (attempt - 1)) % len(CONFIGS)
    order = CONFIGS[shift:] + CONFIGS[:shift]
    for index, (mirror, client, streams) in enumerate(order):
        # A distinct GiB of the way section (from about 64 GB) for each measurement
        start = (60 + 5 * job + index) * (1 << 30)
        line = measure(mirror, client, streams, start)
        print(json.dumps({"kind": "measurement", "job": job, "attempt": attempt,
                          "position": index, **line}), flush=True)


if __name__ == "__main__":
    main()
