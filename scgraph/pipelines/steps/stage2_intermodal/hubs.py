"""Universal intermodal anchoring engine.

Connects multi-modal hubs (seaports, commercial airports, freight rail terminals, dry ports)
to their accessible modal networks under physical medium and topological constraints.
"""

import logging
import math
import os
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scgraph.pipelines.steps.stage3_assembly.assembly import (
    NearNodes,
    PieceArrays,
    build_graph_from_pieces,
    strongly_connected_components,
)
from scgraph.utils import haversine

if TYPE_CHECKING:
    from scgraph.pipelines.steps.stage2_intermodal.landmask import LandMask

logger = logging.getLogger(__name__)

# Default search tolerances in kilometers per mode
DEFAULT_TOLERANCES = {
    "road": 20.0,
    "rail": 10.0,
    "air": 2.0,
    "sea": 2500.0,
}
WATER_MAX_KM = 3.0
ROADSTEAD_WATER_MAX_KM = 10.0
CIRCUITY_LAND = 1.2


@dataclass
class IntermodalHub:
    """An intermodal transfer facility connecting two or more transport modes."""

    latitude: float
    longitude: float
    hub_type: str  # 'port', 'airport', 'rail_terminal', 'dry_port'
    name: str = ""
    capabilities: list[str] = field(
        default_factory=list
    )  # ['road', 'sea', 'rail', 'air']
    harbor_type: str = ""
    dwell_time_hours: float = 2.0
    handling_co2_kg: float = 0.5


def _get_lats_lons(g: dict) -> tuple[Sequence[int], Sequence[int]]:
    """Extract latitude and longitude integer arrays (1e-7 degree) from graph dict."""
    if "latitudes" in g and "longitudes" in g:
        return g["latitudes"], g["longitudes"]
    if "node_points" in g:
        pts = memoryview(g["node_points"])
        return pts[0::2], pts[1::2]
    raise KeyError(
        "Graph missing coordinates (latitudes/longitudes or node_points)"
    )


def join_multimodal_hubs(
    modal_graphs: dict[str, dict[str, array | memoryview]],
    hubs: Iterable[IntermodalHub],
    land_mask: "LandMask",
    *,
    tolerances: dict[str, float] | None = None,
    work_dir: str | os.PathLike | None = None,
) -> tuple[dict, dict[str, int]]:
    """Join multiple modal networks through intermodal hubs.

    Args:
        modal_graphs: Mapping of mode name ('road', 'sea', 'rail', 'air') to graph CSR dicts.
        hubs: Iterable of IntermodalHub instances.
        land_mask: Raster landmask to verify physical medium feasibility.
        tolerances: Maximum snapping distance in km per mode.
        work_dir: Optional intermediate storage directory for PieceArrays.

    Returns:
        Tuple of (combined graph dict, counts of joined entities per mode).
    """
    tols = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    modes_order = [
        m for m in ("road", "sea", "rail", "air") if m in modal_graphs
    ]

    # Pre-index spatial and topological structures for each land mode
    spatial_indices = {}
    components_map = {}
    node_offsets = {}
    total_nodes = 0

    for mode in modes_order:
        g = modal_graphs[mode]
        n_nodes = len(g["offsets"]) - 1
        node_offsets[mode] = total_nodes
        total_nodes += n_nodes

        if mode in ("road", "rail"):
            labels, sizes = strongly_connected_components(
                g["offsets"], g["heads"]
            )
            components_map[mode] = array("i", (sizes[lbl] for lbl in labels))
            lats, lons = _get_lats_lons(g)
            spatial_indices[mode] = NearNodes(lats, lons, scale=1e-7)

    # Prepare sea targets if sea is present
    sea_cells = {}
    sea_targets = array("q")
    sea_nodes = []
    if "sea" in modal_graphs:
        sea_g = modal_graphs["sea"]
        s_lats, s_lons = _get_lats_lons(sea_g)
        for s in range(len(sea_g["offsets"]) - 1):
            pt = [s_lats[s] * 1e-7, s_lons[s] * 1e-7]
            sea_nodes.append(pt)
            cell = land_mask.get_cell(*pt)
            if cell is not None:
                sea_cells.setdefault(cell, s)
        sea_targets = array("q", sorted(sea_cells))

    # Prepare air targets if air is present
    if "air" in modal_graphs:
        air_g = modal_graphs["air"]
        a_lats, a_lons = _get_lats_lons(air_g)
        spatial_indices["air"] = NearNodes(a_lats, a_lons, scale=1e-7)

    pieces = PieceArrays(tolerance_meters=None, work_dir=work_dir)

    # Copy modal internal arcs
    for mode in modes_order:
        g = modal_graphs[mode]
        base_u = node_offsets[mode]
        offsets, heads, weights = g["offsets"], g["heads"], g["weights"]
        starts, line = g["shape_offsets"], g["shape_points"]
        lats, lons = _get_lats_lons(g)

        for u in range(len(offsets) - 1):
            pt_u = [lats[u] * 1e-7, lons[u] * 1e-7]
            for a in range(offsets[u], offsets[u + 1]):
                v = heads[a]
                pt_v = [lats[v] * 1e-7, lons[v] * 1e-7]
                shape_slice = line[2 * starts[a] : 2 * starts[a + 1]]
                shape = [
                    [x * 1e-7, y * 1e-7]
                    for x, y in zip(shape_slice[::2], shape_slice[1::2])
                ]
                pieces.append(
                    base_u + u, base_u + v, weights[a], 1, [pt_u, pt_v], shape
                )

    # Connect hubs
    hub_nodes = []
    hub_start_id = total_nodes
    joined_counts = {"hubs": 0}

    for hub in hubs:
        point = [hub.latitude, hub.longitude]
        connectors_by_mode = {}

        # Check Road
        if "road" in hub.capabilities and "road" in spatial_indices:
            conn = _find_land_connector(
                point,
                spatial_indices["road"],
                components_map["road"],
                land_mask,
                tols["road"],
                is_roadstead=(hub.harbor_type == "Open Roadstead"),
            )
            if conn:
                connectors_by_mode["road"] = conn

        # Check Rail
        if "rail" in hub.capabilities and "rail" in spatial_indices:
            conn = _find_land_connector(
                point,
                spatial_indices["rail"],
                components_map["rail"],
                land_mask,
                tols["rail"],
                is_roadstead=False,
            )
            if conn:
                connectors_by_mode["rail"] = conn

        # Check Air
        if "air" in hub.capabilities and "air" in spatial_indices:
            air_idx = spatial_indices["air"]
            candidates = air_idx.find_within(point, tols["air"])
            if candidates:
                best_km, best_u = min(candidates)
                connectors_by_mode["air"] = (best_u, best_km, [])

        # Check Sea
        if "sea" in hub.capabilities and sea_targets:
            sea_conn = _find_sea_connector(
                point, sea_targets, sea_cells, sea_nodes, land_mask, tols["sea"]
            )
            if sea_conn:
                connectors_by_mode["sea"] = sea_conn

        # Admissibility rule: Must connect at least TWO distinct transport modes
        if len(connectors_by_mode) >= 2:
            current_hub_id = hub_start_id + len(hub_nodes)
            hub_nodes.append(point)
            joined_counts["hubs"] += 1

            for mode, conn in connectors_by_mode.items():
                target_u, km, shape = (
                    conn[0],
                    conn[1],
                    conn[2] if len(conn) > 2 else [],
                )
                global_target_u = node_offsets[mode] + target_u
                circuity = CIRCUITY_LAND if mode in ("road", "rail") else 1.0
                weighted_km = km * circuity
                target_pt = _get_node_pt(modal_graphs[mode], target_u)

                # Bidirectional connector arcs
                pieces.append(
                    current_hub_id,
                    global_target_u,
                    weighted_km,
                    1,
                    [point, target_pt],
                    shape,
                )
                pieces.append(
                    global_target_u,
                    current_hub_id,
                    weighted_km,
                    1,
                    [target_pt, point],
                    shape[::-1],
                )

    pieces.flush()
    joined_graph = build_graph_from_pieces(pieces)

    # Build node_modes array
    # Modes: road=0, sea=1, rail=2, air=3, hub=4
    mode_indices = {m: i for i, m in enumerate(modes_order)}
    hub_mode_index = len(modes_order)
    node_modes_raw = bytearray()

    for mode in modes_order:
        n_nodes = len(modal_graphs[mode]["offsets"]) - 1
        node_modes_raw.extend(bytes([mode_indices[mode]]) * n_nodes)
    node_modes_raw.extend(bytes([hub_mode_index]) * len(hub_nodes))

    joined_graph["node_modes"] = array(
        "B", (node_modes_raw[u] for u in joined_graph["node_ids"])
    )

    # Assign node IDs (source IDs for modal nodes, negative for hubs)
    all_ids = array("q")
    for mode in modes_order:
        all_ids.extend(
            modal_graphs[mode].get(
                "node_ids",
                array("q", range(len(modal_graphs[mode]["offsets"]) - 1)),
            )
        )
    all_ids.extend(array("q", range(-1, -1 - len(hub_nodes), -1)))
    joined_graph["node_ids"] = array(
        "q", (all_ids[u] for u in joined_graph["node_ids"])
    )

    return joined_graph, joined_counts


def _find_land_connector(
    point: list[float],
    spatial_index: NearNodes,
    component_sizes: Sequence[int],
    land_mask: "LandMask",
    max_km: float,
    is_roadstead: bool = False,
) -> tuple[int, float, list] | None:
    """Find the best land connector (road or rail) avoiding water crossing."""
    water_threshold = ROADSTEAD_WATER_MAX_KM if is_roadstead else WATER_MAX_KM
    candidates = spatial_index.find_within(point, max_km)
    if not candidates:
        return None

    # Largest component first, then nearest
    for _, km, i in sorted(
        (-component_sizes[i], km, i) for km, i in candidates
    ):
        target = [
            spatial_index.lats[i] * spatial_index.scale,
            spatial_index.lons[i] * spatial_index.scale,
        ]
        if _water_km(land_mask, point, target, km) < water_threshold:
            return i, km, []
    return None


def _find_sea_connector(
    port_coords: Sequence[float],
    sea_targets: array,
    cell_to_sea_node: dict[int, int],
    sea_nodes: Sequence[Sequence[float]],
    land_mask: "LandMask",
    max_km: float,
) -> tuple[int, float, list[list[float]]] | None:
    """Find sea connector via A* water path."""
    found = land_mask.find_water_path(port_coords, sea_targets, max_km)
    if not found:
        return None
    km, path = found
    s = cell_to_sea_node[land_mask.get_cell(*path[-1])]
    return s, km + haversine(path[-1], sea_nodes[s]), path


def _water_km(land_mask, pt_a, pt_b, distance_km: float) -> float:
    """Kilometers of water along straight line pt_a to pt_b."""
    n = max(1, math.ceil(distance_km / 0.25))
    water = sum(
        not land_mask.is_on_land(
            *(x + (j + 0.5) / n * (y - x) for x, y in zip(pt_a, pt_b))
        )
        for j in range(n)
    )
    return distance_km * water / n


def _get_node_pt(graph: dict, u: int) -> list[float]:
    """Retrieve [latitude, longitude] of node u from graph."""
    lats, lons = _get_lats_lons(graph)
    return [lats[u] * 1e-7, lons[u] * 1e-7]
