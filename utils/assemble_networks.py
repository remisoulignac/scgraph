"""Assemble a data release from the networks built by the workflow `build_networks.yml`.

Usage: python utils/assemble_networks.py BUILT PREVIOUS RELEASE TAG COMMIT

BUILT holds the network files of this build. PREVIOUS holds the files of the previous
data release, or nothing. The script writes into RELEASE one file per network, the index
`networks.json` and the notes `notes.md`.

A network is carried over when its pipeline and its sources did not change: the release
then keeps the file of the previous release, byte for byte. The URL of a source is left
out of the comparison, because the same file can be read from several mirrors.
"""

import json
import shutil
import sys
from pathlib import Path


def read_header(filename: Path) -> dict:
    """Return the metadata of a network file, with its `format_version` and its
    `content_sha256`."""
    with open(filename, "rb") as file:
        if file.read(8) != b"SCGRAPH1":
            raise ValueError(f"{filename} is not a scgraph network file")
        digest = file.read(32).hex()
        header = json.loads(file.read(int.from_bytes(file.read(8), "little")))
    return {
        **header["metadata"],
        "format_version": header["format_version"],
        "content_sha256": digest,
    }


def get_build_inputs(header: dict) -> tuple:
    """Return what decides the graph of a built network: its pipeline and its sources."""
    sources = [
        {key: value for key, value in source.items() if key != "url"}
        for source in header["sources"]
    ]
    return header["pipeline"], sources


def main(built, previous, release, tag, commit):
    release = Path(release)
    release.mkdir()
    networks = []
    for file in sorted(Path(built).glob("*.scgraph")):
        header = read_header(file)
        earlier = Path(previous, file.name)
        is_carried_over = earlier.is_file() and get_build_inputs(
            read_header(earlier)
        ) == get_build_inputs(header)
        kept = earlier if is_carried_over else file
        shutil.copyfile(kept, release / file.name)
        networks.append(
            {
                "file": file.name,
                "name": header["pipeline"]["name"],
                "region": header["pipeline"]["region"],
                "status": "carried over" if is_carried_over else "rebuilt",
                "bytes": kept.stat().st_size,
                "content_sha256": read_header(kept)["content_sha256"],
                "built_content_sha256": header["content_sha256"],
                "format_version": header["format_version"],
                "pipeline_version": header["pipeline"]["version"],
                "sources": header["sources"],
            }
        )
    index = {"tag": tag, "commit": commit, "networks": networks}
    (release / "networks.json").write_text(json.dumps(index, indent=1) + "\n")
    rows = [
        f"| `{n['file']}` | {n['status']} | {n['bytes'] / 1e6:.1f} MB "
        f"| `{n['content_sha256']}` |"
        for n in networks
    ]
    (release / "notes.md").write_text(
        f"Built from commit {commit}.\n\n"
        "| File | Status | Size | `content_sha256` |\n|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
