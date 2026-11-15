"""Global 4-Mode Multimodal Pipeline (Road, Sea, Rail, Air).

Combines:
- Main roads and ferries from OpenStreetMap
- Sea lanes from Eurostat SeaRoute marnet
- Railway network from Natural Earth 1:10m Railroads
- Commercial air transport from OurAirports and OpenFlights routes
- Intermodal transfers via universal hubs (seaports, commercial airports, rail terminals)

Supports duration-CO2 Pareto frontier routing via parametric edge cost scalarization.
"""

import os
from array import array
from collections.abc import Callable

from scgraph.pipelines import eurostat_marnet
from scgraph.pipelines.steps.stage1_extract import air, osm, rail
from scgraph.pipelines.steps.stage2_intermodal import hubs, ports
from scgraph.pipelines.steps.stage2_intermodal.hubs import IntermodalHub
from scgraph.pipelines.steps.stage2_intermodal.landmask import LandMask
from scgraph.pipelines.steps.stage3_assembly import assembly

PIPELINE_VERSION = 1


def _natural_earth(layer: str) -> dict:
    return {
        "url": (
            f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            f"f1890d9f152c896d250a77557a5751a93d494776/geojson/ne_10m_{layer}.geojson"
        ),
        "credit": "Natural Earth (1:10m)",
        "version": "5.1.2",
        "license": "CC0-1.0",
        "license_url": "https://github.com/nvkelso/natural-earth-vector/blob/master/LICENSE.md",
    }


SOURCES = {
    "roads": osm.PLANET,
    "sea": eurostat_marnet.SOURCES["sea_lanes"],
    "rail": rail.NATURAL_EARTH_RAIL,
    "airports": air.OURAIRPORTS,
    "flights": air.OPENFLIGHTS_ROUTES,
    "ports": {
        "url": (
            "https://msi.nga.mil/api/publications/download?type=view&"
            "key=16920959/SFH00000/UpdatedPub150.csv"
        ),
        "download": ports.download,
        "credit": "National Geospatial-Intelligence Agency, World Port Index (Pub. 150)",
        "license": "Public Domain",
        "license_url": "https://msi.nga.mil/Publications/WPI",
    },
    "land": _natural_earth("land"),
    "lakes": _natural_earth("lakes"),
    "rivers": _natural_earth("rivers_lake_centerlines"),
}

MAX_KM = 10.0
MIN_NODES = 100
MARGIN = 2.0  # degrees of raster margin

SELECTS = {
    "roads": osm.select_main_roads,
    "ferries": osm.select_vehicle_ferries,
}

MODES = [
    {"name": "road", "source": "roads"},
    {"name": "sea", "source": "sea"},
    {"name": "rail", "source": "rail"},
    {"name": "air", "source": "flights"},
    {"name": "hub"},
]

download = osm.download


def build(
    files: dict[str, str],
    *,
    source: Callable = osm.open_osm_source,
    max_km_between_nodes: float = MAX_KM,
    min_nodes_per_strongly_connected_component: int | None = MIN_NODES,
    work_dir: str | os.PathLike | None = None,
) -> tuple[tuple[dict, list[dict]], list[dict]]:
    """Build the global 4-mode multimodal transport network.

    Args:
        files: Mapping of source identifiers to local file paths.
        source: Factory resolving the OSM source reader.
        max_km_between_nodes: Spacing in km for degree-2 contraction.
        min_nodes_per_strongly_connected_component: Minimum SCC size threshold.
        work_dir: Intermediate scratch directory.

    Returns:
        Tuple of ((arrays, MODES), steps) describing the built multimodal network.
    """
    steps = []

    # 1. Extract Roads
    road_pieces, road_steps = osm.build_way_pieces(
        source(files["roads"]),
        SELECTS,
        ("highway", "route"),
        max_km_between_nodes,
        os.path.join(work_dir, "roads") if work_dir else None,
    )
    road_graph = assembly.build_graph_from_pieces(road_pieces)
    steps.extend(road_steps)

    # 2. Extract Sea
    sea_geo = eurostat_marnet.build({"sea_lanes": files["sea"]})
    sea_graph = _geograph_to_csr_dict(sea_geo)
    steps.append({"step": "sea", "nodes": len(sea_geo.nodes)})

    # 3. Extract Rail
    rail_pieces, rail_steps = rail.extract_rail_pieces(
        files["rail"],
        source_type="natural_earth",
        max_km_between_nodes=max_km_between_nodes,
        work_dir=os.path.join(work_dir, "rail") if work_dir else None,
    )
    rail_graph = assembly.build_graph_from_pieces(rail_pieces)
    steps.extend(rail_steps)

    # 4. Extract Air
    airports_data = air.read_airports(files["airports"])
    flight_pieces, _, air_steps = air.extract_flight_pieces(
        files["flights"],
        airports_data,
        work_dir=os.path.join(work_dir, "air") if work_dir else None,
    )
    air_graph = assembly.build_graph_from_pieces(flight_pieces)
    steps.extend(air_steps)

    # 5. Build LandMask
    node_points = memoryview(road_graph["node_points"])
    lats, lons = node_points[0::2], node_points[1::2]
    south = max(-90.0, min(lats) * 1e-7 - MARGIN)
    west = max(-180.0, min(lons) * 1e-7 - MARGIN)
    north = min(90.0, max(lats) * 1e-7 + MARGIN)
    east = min(180.0, max(lons) * 1e-7 + MARGIN)
    del node_points, lats, lons

    mask = LandMask.from_natural_earth(
        *(files[name] for name in ("land", "lakes", "rivers")),
        bbox=(south, west, north, east),
    )

    # 6. Assemble Universal Intermodal Hubs
    hub_records = []

    # Seaports (from WPI)
    for p in ports.read_world_port_index(files["ports"]):
        if mask.get_cell(*p[:2]) is not None:
            caps = ["sea", "road"]
            hub_records.append(
                IntermodalHub(
                    latitude=p[0],
                    longitude=p[1],
                    hub_type="port",
                    capabilities=caps,
                    harbor_type=p[2],
                    dwell_time_hours=4.0,
                    handling_co2_kg=0.8,
                )
            )

    # Airports (from OurAirports commercial hubs)
    for apt in airports_data.values():
        hub_records.append(
            IntermodalHub(
                latitude=apt["latitude"],
                longitude=apt["longitude"],
                hub_type="airport",
                name=apt["name"],
                capabilities=["air", "road", "rail"],
                dwell_time_hours=2.0,
                handling_co2_kg=1.2,
            )
        )

    # 7. Join Modal Graphs through Hubs
    modal_graphs = {
        "road": road_graph,
        "sea": sea_graph,
        "rail": rail_graph,
        "air": air_graph,
    }

    joined_graph, counts = hubs.join_multimodal_hubs(
        modal_graphs,
        hub_records,
        mask,
        work_dir=os.path.join(work_dir, "hubs") if work_dir else None,
    )
    steps.append({"step": "hubs_joined", **counts})

    # 8. Assemble & Hilbert Order
    arrays, assembled = assembly.assemble(
        joined_graph,
        max_km_between_nodes=max_km_between_nodes,
        min_nodes_per_strongly_connected_component=min_nodes_per_strongly_connected_component,
    )
    return (arrays, MODES), steps + assembled


def _geograph_to_csr_dict(geo) -> dict:
    """Convert an unreduced GeoGraph into a CSR graph dict."""
    offsets, heads, weights = assembly.build_csr_from_adjacency_list(
        geo.graph_object.graph
    )
    pts = []
    for lat, lon in geo.nodes:
        pts.extend([round(lat * 1e7), round(lon * 1e7)])
    shape_offsets = array("i", range(len(heads) + 1))
    shape_points = array("i")
    return {
        "offsets": offsets,
        "heads": heads,
        "weights": weights,
        "node_points": array("i", pts),
        "shape_offsets": shape_offsets,
        "shape_points": shape_points,
        "node_ids": array("q", range(len(geo.nodes))),
    }


VALIDATION = {
    "cases": [
        "road_distances",
        "sea_distances",
        "rail_corridors",
        "passages",
        "river_ports",
    ]
}
