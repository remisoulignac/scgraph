"""Assemble a data release from the networks that the workflow `build_networks.yml` built.

Usage: python utils/assemble_networks.py BUILT PREVIOUS RELEASE TAG COMMIT

A data release is a set of network files published as the assets of one GitHub release.
BUILT holds the network files of this build. PREVIOUS holds the files of the previous
data release, or nothing. The script writes into RELEASE:

- one file per network. A network is carried over when its pipeline and its sources
  did not change since the previous release: the release then keeps the previous file,
  byte for byte. Otherwise the release takes the new file;
- `networks.json`, the index of the release: the tag, the commit that built the
  networks, and one entry per network;
- `notes.md`, the notes of the release, with the credit and the license of each network.

The script stops when a source of a network has no established license.
"""

import json
import shutil
import sys
from pathlib import Path

from scgraph import GeoGraph


def get_build_inputs(info: dict) -> tuple[dict, list[dict]]:
    """Return what decides the graph of a built network: its pipeline and its sources.

    Args:
        info: What `GeoGraph.get_scgraph_info` returns for the network file.

    Returns:
        The pipeline (name, region, version), and the record of each source without
        its URL. The same source can be read from several mirrors, so its URL does not
        identify it: its digest and its version do.
    """
    sources = [
        {key: value for key, value in source.items() if key != "url"}
        for source in info["sources"]
    ]
    return info["pipeline"], sources


def assemble(
    built: str, previous: str, release: str, tag: str, commit: str
) -> None:
    """Write the files, the index and the notes of the data release `tag` into the
    new folder `release`.

    Args:
        built: The folder of the network files of this build.
        previous: The folder of the files of the previous data release. It may be empty.
        release: The folder to create.
        tag: The tag of the release.
        commit: The commit that built the networks.

    Raises:
        ValueError: A source of a network has no license, or `NOASSERTION`.
    """
    folder = Path(release)
    folder.mkdir()
    networks = []
    for file in sorted(Path(built).glob("*.scgraph")):
        info = GeoGraph.get_scgraph_info(file)
        for source in info["sources"]:
            if source.get("license", "NOASSERTION") == "NOASSERTION":
                raise ValueError(
                    f"{file.name}: the source '{source['name']}' has no "
                    "established license, so the network is not published"
                )
        earlier = Path(previous, file.name)
        is_carried_over = earlier.is_file() and get_build_inputs(
            GeoGraph.get_scgraph_info(earlier)
        ) == get_build_inputs(info)
        if is_carried_over:
            file, info = earlier, GeoGraph.get_scgraph_info(earlier)
        shutil.copyfile(file, folder / file.name)
        networks.append(
            {
                "file": file.name,
                "status": "carried over" if is_carried_over else "rebuilt",
                "bytes": file.stat().st_size,
                **{
                    key: info[key]
                    for key in (
                        "content_sha256",
                        "format_version",
                        "nodes",
                        "arcs",
                        "data_up_to",
                        "attribution",
                        "pipeline",
                        "sources",
                    )
                },
            }
        )
    index = {"tag": tag, "commit": commit, "networks": networks}
    (folder / "networks.json").write_text(json.dumps(index, indent=1) + "\n")
    (folder / "notes.md").write_text(write_notes(index))


def write_notes(index: dict) -> str:
    """Return the notes of a data release, in Markdown: how to load a file, then one
    row per network, then the credit and the license of each network."""
    rows = [
        f"| `{n['file']}` | {n['bytes'] / 1e6:.1f} MB | {n['nodes']:,} "
        f"| {n['arcs']:,} | {(n['data_up_to'] or '')[:10] or 'no date'} "
        f"| {n['status']} | `{n['content_sha256'][:16]}` |"
        for n in index["networks"]
    ]
    credits = [
        f"- `{n['file']}`: {n['attribution']}" for n in index["networks"]
    ]
    return "\n".join(
        [
            "Network files built by the pipelines of scgraph, from commit "
            f"{index['commit']}. Each network passed the validation cases of its "
            "pipeline.",
            "",
            "To use a file, install scgraph from the `dev` branch, download the "
            'file, and load it with `GeoGraph.load_from_scgraph("FILE")`. '
            '`GeoGraph.get_scgraph_info("FILE")` gives its sources and its '
            "`content_sha256`. `networks.json` holds the full digests.",
            "",
            "| File | Size | Nodes | Arcs | Data up to | Status | `content_sha256` |",
            "|---|---|---|---|---|---|---|",
            *rows,
            "",
            "**Credits and licenses.** Cite the credit line of a network with "
            "the results that use it.",
            "",
            *credits,
            "",
        ]
    )


if __name__ == "__main__":
    assemble(*sys.argv[1:])
