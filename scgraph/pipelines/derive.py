"""Derive a network file from another network file, with no download.

WHAT IT ANSWERS

The network `world_multimodal` holds the roads, the railways, the sea lanes and the
flights of the world. A smaller network is a part of it: some of its modes, or its
land networks in one region. `derive_file` takes a network file, the "parent file".
It writes such a part as a new network file, the "derived file". A "view" is a named
choice of modes and of region. `VIEWS` lists the views of `world_multimodal` for a
data release.

A route of the derived file is a route of the parent file, with the same length.

WHAT IT READS

A pipeline with several modes built the parent file, for example `world_multimodal`.
The file holds the mode of each node and the shape of each arc. Its coordinates are
in units of 1e-7 degree. `derive_file` maps the parent file, and copies into memory
only the part that it keeps.

HOW A RUN GOES

With every mode and no rectangle, nothing is cut: the run goes to step 4 with the
network of the parent file.

1. Select the nodes. A node is selected when its mode is one of `modes`. A node of a
   land mode (`LAND_MODES`) must also be inside `rectangle`. The rectangle does not
   cut the other modes: their nodes all stay.
2. Cut the graph. An arc stays when its two ends are selected.
3. Assemble the graph again, with `assembly.assemble`. This function keeps each
   strongly connected component of at least `min_component_nodes` nodes. It contracts
   the chains: a junction that lost a road in step 2 can now be a pass-through node.
   It numbers the nodes by mode, then along a Hilbert curve.
4. Write the derived file. Its modes table holds the kept modes only. As in every
   network file, a derived file with one mode has no `node_modes`.

Example: a road joins the nodes 1, 2 and 3 along the equator, with 3 km between two
nodes. A second road goes from node 2 to node 4, which is two degrees north. A port
joins node 3 to a sea lane that ends at 50 degrees east. The rectangle (-1, 1, -1, 1)
removes node 4, and the sea lane stays whole. Node 2 is then between the nodes 1 and
3 only, so step 3 contracts it. The derived file has one arc of 6 km from node 1 to
node 3.

WHY THE RESULT IS RIGHT

Steps 1 and 2 only remove nodes and arcs. Each arc that stays is an arc of the parent
file, with its length. Step 3 does not change the length of a route between two kept
nodes (`assembly`, *Why the result is right*). So a route of the derived file has the
length of the same route in the parent file. The shortest route of the parent file
is shorter only when it passes through a node that the derived file does not hold.

WHAT THE HEADER OF A DERIVED FILE RECORDS

- `pipeline` and `description`: those of the parent file;
- `sources`: the sources of the parent file that serve the derived file. A mode cites
  the source of its node ids in the modes table. A source leaves when only removed
  modes cite it. A source that no mode cites stays: the header does not say which
  mode it serves;
- `software`: the software that derived the file;
- `derivation`: the `parent_content_sha256`, and the `modes`, the `rectangle` and the
  `min_component_nodes` of the call.

HOW A DERIVED FILE IS VALIDATED

`validate_file` checks a derived file against the validation cases of the pipeline of
its parent file. It checks a case when two conditions hold. The two ends of the case
are inside the rectangle. The derived file holds the modes on which the kind of the
case searches its routes (`RULES` of `stage4_validation`). Some kinds search on every
mode: the derived file must then hold every mode of the parent file.

THE COMMAND LINE

`python -m scgraph.pipelines.derive PARENT VIEW OUTPUT_DIR` writes
`OUTPUT_DIR/VIEW.scgraph` and prints its path. `--help` lists its two options.
"""

import argparse
import logging
import os
from array import array
from collections.abc import Sequence

from scgraph import network_file, pipelines
from scgraph.pipelines import world_multimodal
from scgraph.pipelines.build import _resolve_pipeline, _validate_pipeline
from scgraph.pipelines.steps.stage3_assembly import assembly
from scgraph.pipelines.steps.stage4_validation import RULES
from scgraph.utils import _log_step

logger = logging.getLogger(__name__)

# The modes that a rectangle cuts. The nodes of the other modes (`sea`, `port`,
# `airport`) all stay. A route can then leave the region by sea or by air, and reach
# a port or an airport outside it.
LAND_MODES = ("road", "rail", "rail_terminal")
# The rectangle of a call without a rectangle: every point is inside it.
_WHOLE_GLOBE = (-90, 90, -180, 180)
# The views of `world_multimodal` for a data release, as `name: (modes, rectangle)`.
# These are the arguments of `derive_file`: None keeps every mode, and None cuts
# nothing. A rectangle is (latitude min, latitude max, longitude min, longitude max),
# in degrees. The view `world_multimodal` cuts nothing: it gives the parent file
# without the shapes of its arcs.
VIEWS: dict[str, tuple[list[str] | None, Sequence[float] | None]] = {
    "world_multimodal": (None, None),
    "world_sea": (["sea", "port"], None),
    "world_rail": (["rail", "rail_terminal"], None),
    "europe": (None, (34, 72, -25, 45)),
    "north_america": (None, (7, 84, -170, -50)),
    "south_america": (None, (-57, 13, -82, -34)),
    "africa": (None, (-35, 38, -18, 52)),
    "west_south_asia": (None, (5, 45, 25, 98)),
    "east_southeast_asia": (None, (-11, 55, 92, 150)),
    "north_asia": (None, (40, 78, 45, 180)),
    "oceania": (None, (-50, 0, 110, 180)),
}


def derive_file(
    parent_file: str | os.PathLike,
    derived_file: str | os.PathLike,
    modes: list[str] | None = None,
    rectangle: Sequence[float] | None = None,
    *,
    keep_shapes: bool = True,
    min_component_nodes: int = 10,
) -> str:
    """Write a part of a network file as a new network file: some of its modes, and
    its land networks inside a rectangle. The module docstring gives the steps.

    Args:
        parent_file: The network file to derive from. It holds several modes and the
            shapes of its arcs.
        derived_file: The path of the network file to write.
        modes: The names of the modes to keep. None keeps every mode.
        rectangle: `(latitude min, latitude max, longitude min, longitude max)`, in
            degrees. A node of `LAND_MODES` outside the rectangle is removed. None
            removes no node.
        keep_shapes: False writes the same network without the shapes of its arcs.
        min_component_nodes: The smallest number of nodes of a kept strongly
            connected component. The default removes a node that the cut left
            alone, and keeps the roads of a small island.

    Returns:
        The `content_sha256` of the derived file.

    Raises:
        ValueError: The parent file has one mode only, or no arc shapes. Or `modes`
            names a mode that the parent file does not have. Or no node of these
            modes is inside the rectangle.
    """
    arrays, parent = network_file.read(parent_file, mapped=True)
    if not {"node_modes", "shape_points"} <= set(arrays):
        raise ValueError(
            f"{parent_file}: a parent file holds several modes and arc shapes"
        )
    names = [mode["name"] for mode in parent["modes"]]
    unknown = sorted(set(modes or ()) - set(names))
    if unknown:
        raise ValueError(f"{parent_file}: unknown modes {unknown}")
    # The rows of the modes table of the parent file that the derived file keeps
    kept_rows = [
        row for row, name in enumerate(names) if modes is None or name in modes
    ]
    # With every mode and no rectangle, nothing is cut: the network is the parent's
    if modes is not None or rectangle is not None:
        arrays = _cut(
            parent_file,
            arrays,
            names,
            kept_rows,
            rectangle,
            min_component_nodes,
        )
    # 4. Write the derived file
    if len(kept_rows) == 1:  # a network file with one mode has no `node_modes`
        del arrays["node_modes"]
    else:  # a kept mode takes its row in the new modes table
        new_rows = bytes.maketrans(
            bytes(kept_rows), bytes(range(len(kept_rows)))
        )
        arrays["node_modes"] = array(
            "B", arrays["node_modes"].tobytes().translate(new_rows)
        )
    if not keep_shapes:
        del arrays["shape_offsets"], arrays["shape_points"]
    kept_modes = [parent["modes"][row] for row in kept_rows]
    removed_sources = {mode.get("source") for mode in parent["modes"]} - {
        mode.get("source") for mode in kept_modes
    }
    metadata = pipelines.build_metadata(
        **parent["pipeline"],
        sources=[
            source
            for source in parent["sources"]
            if source["name"] not in removed_sources
        ],
        description=parent.get("description"),
    )
    metadata["derivation"] = {
        "parent_content_sha256": parent["content_sha256"],
        "modes": modes,
        "rectangle": rectangle,
        "min_component_nodes": min_component_nodes,
    }
    return network_file.write(
        derived_file,
        arrays,
        network_file.header_metadata(
            metadata, units=parent["units"], modes=kept_modes
        ),
    )


def _cut(parent_file, arrays, names, kept_rows, rectangle, min_component_nodes):
    # Steps 1 to 3 of the module docstring, on the arrays of the parent file
    with _log_step(
        logger, "select the nodes", assembly._count_nodes_and_arcs(arrays)
    ) as counts:
        # 1. Select the nodes. The rectangle is in the unit of the node points.
        land_rows = {row for row in kept_rows if names[row] in LAND_MODES}
        south, north, west, east = (
            round(degrees * 1e7) for degrees in rectangle or _WHOLE_GLOBE
        )
        latitudes, longitudes = (arrays["node_points"][i::2] for i in (0, 1))
        selected = array(
            "i",
            (
                u
                for u, row in enumerate(arrays["node_modes"])
                if row in kept_rows
                and (
                    row not in land_rows
                    or south <= latitudes[u] <= north
                    and west <= longitudes[u] <= east
                )
            ),
        )
        # 2. Cut the graph
        if not selected:
            raise ValueError(
                f"{parent_file}: no node of these modes is inside the rectangle"
            )
        graph = assembly._renumbered(arrays, selected)
        counts.update(assembly._count_nodes_and_arcs(graph, "kept_"))
    # 3. Assemble the graph again, with the node spacing of the world network
    return assembly.assemble(
        graph,
        max_km_between_nodes=world_multimodal.MAX_KM_BETWEEN_NODES,
        min_component_nodes=min_component_nodes,
    )


def validate_file(derived_file: str | os.PathLike) -> None:
    """Check a derived file against the validation cases of the pipeline of its
    parent file. The header of the derived file gives the pipeline, the modes and
    the rectangle. The module docstring says which cases are checked.

    Raises:
        RuntimeError: A case fails.
    """
    info = network_file.get_scgraph_info(derived_file)
    built_by, derivation = info["pipeline"], info["derivation"]
    modes = derivation["modes"]
    south, north, west, east = derivation["rectangle"] or _WHOLE_GLOBE

    def should_check(case):
        # A kind with an empty tuple of modes searches its routes on every mode
        needed = RULES[case["kind"]][0]
        return (
            modes is None or (bool(needed) and set(needed) <= set(modes))
        ) and all(
            south <= latitude <= north and west <= longitude <= east
            for latitude, longitude in (case["origin"], case["destination"])
        )

    pipeline, _ = _resolve_pipeline(built_by["name"])
    _validate_pipeline(
        pipeline,
        str(derived_file),
        built_by["region"],
        derived_file,
        should_check,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m scgraph.pipelines.derive",
        description="Derive a published view from a network file, with no "
        "download, and print the path of the file written.",
    )
    parser.add_argument("parent", help="the network file to derive from")
    parser.add_argument("view", choices=VIEWS, help="the view to derive")
    parser.add_argument("output_dir", help="the folder of the derived files")
    parser.add_argument(
        "--without-shapes",
        action="store_true",
        help="write VIEW.no_shapes.scgraph: the same network without the shapes "
        "of its arcs",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="check the derived file against the validation cases of the "
        "pipeline of the parent file; a file that fails stays in the folder",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)  # the count of validation cases
    suffix = ".no_shapes" if args.without_shapes else ""
    path = os.path.join(args.output_dir, f"{args.view}{suffix}.scgraph")
    try:
        derive_file(
            args.parent,
            path,
            *VIEWS[args.view],
            keep_shapes=not args.without_shapes,
        )
        if args.validate:
            validate_file(path)
        print(path)  # noqa: T201 (the output of the command)
    except (
        ValueError,
        FileNotFoundError,
        RuntimeError,
    ) as error:  # the message says what to change: no traceback
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
