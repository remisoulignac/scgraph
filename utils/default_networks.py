"""The default data release of this version of scgraph: write it, or check it.

Usage:
    python utils/default_networks.py pin TAG
    python utils/default_networks.py check

`pin` reads the index `networks.json` of the data release TAG and writes its tag and the
digest of each file into `utils/default_networks.json`. `check` fails when that file and
the published release do not agree. The workflow `build_wheels.yml` runs `check` before
it builds the wheels.
"""

import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT = ROOT / "utils" / "default_networks.json"
REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "connor-makowski/scgraph")


def read_index(tag: str) -> dict:
    """Return the index of the published data release `tag`. A draft has no public
    address, so it fails as a tag that does not exist."""
    url = (
        f"https://github.com/{REPOSITORY}/releases/download/{tag}/networks.json"
    )
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        sys.exit(
            f"{tag} is not a published data release of {REPOSITORY}: {error}"
        )


def pin(tag: str) -> None:
    files = {
        n["file"]: n["content_sha256"] for n in read_index(tag)["networks"]
    }
    DEFAULT.write_text(
        json.dumps({"tag": tag, "files": files}, indent=1) + "\n"
    )


def check() -> None:
    default = json.loads(DEFAULT.read_text())
    tag = default["tag"]
    networks = read_index(tag)["networks"]
    published = {n["file"]: n["content_sha256"] for n in networks}
    if published != default["files"]:
        sys.exit(f"The digests of {DEFAULT.name} differ from those of {tag}")
    reader = (ROOT / "scgraph" / "network_file.py").read_text()
    read_version = int(re.search(r"^VERSION = (\d+)$", reader, re.M)[1])
    if any(n["format_version"] > read_version for n in networks):
        sys.exit(f"{tag} holds a file of a format version above {read_version}")
    with open(ROOT / "pyproject.toml", "rb") as file:
        version = tomllib.load(file)["project"]["version"]
    if not re.search("[a-zA-Z]", version) and "-dev." in tag:
        sys.exit(
            f"The final version {version} names the development release {tag}"
        )
    print(
        f"scgraph {version} loads {tag} by default: {len(published)} files agree"
    )


if __name__ == "__main__":
    pin(sys.argv[2]) if sys.argv[1] == "pin" else check()
