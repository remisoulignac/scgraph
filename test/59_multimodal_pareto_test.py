"""Tests for multimodal data pipelines (rail, air, universal hubs) and Pareto routing."""

import json

from scgraph.geograph import GeoGraph
from scgraph.pipelines.steps.stage1_extract import air, rail
from scgraph.pipelines.steps.stage2_intermodal import hubs
from scgraph.pipelines.steps.stage2_intermodal.hubs import IntermodalHub
from scgraph.pipelines.steps.stage2_intermodal.landmask import LandMask
from scgraph.pipelines.steps.stage3_assembly import assembly


def test_rail_extraction_from_geojson(tmp_path):
    """Verify that railway lines are correctly parsed and cut from GeoJSON."""
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [2.3522, 48.8566],  # Paris [lon, lat]
                        [3.0573, 50.6292],  # Lille
                        [4.3517, 50.8503],  # Brussels
                    ],
                },
                "properties": {"name": "Paris-Brussels High Speed Rail"},
            }
        ],
    }
    file_path = tmp_path / "rail.geojson"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f)

    pieces, steps = rail.extract_rail_pieces(
        file_path,
        source_type="natural_earth",
        max_km_between_nodes=100.0,
    )
    assert len(pieces.km) > 0
    assert steps[0]["step"] == "read_rail"
    assert steps[0]["lines"] == 1


def test_air_extraction_and_flight_routes(tmp_path):
    """Verify OurAirports and OpenFlights extraction."""
    airports_csv = tmp_path / "airports.csv"
    with open(airports_csv, "w", encoding="utf-8") as f:
        f.write(
            "id,ident,type,name,latitude_deg,longitude_deg,scheduled_service,iata_code\n"
            "1,LFPG,large_airport,Paris CDG,49.0097,2.5479,yes,CDG\n"
            "2,EDDF,large_airport,Frankfurt,50.0379,8.5622,yes,FRA\n"
            "3,EGLL,large_airport,London Heathrow,51.4700,-0.4543,yes,LHR\n"
        )

    routes_dat = tmp_path / "routes.dat"
    with open(routes_dat, "w", encoding="utf-8") as f:
        f.write("AF,137,CDG,1,FRA,2,,0,320\n" "BA,135,LHR,3,CDG,1,,0,320\n")

    airports = air.read_airports(airports_csv)
    assert len(airports) == 3
    assert "CDG" in airports
    assert airports["CDG"]["name"] == "Paris CDG"

    flight_pieces, code_to_id, steps = air.extract_flight_pieces(
        routes_dat, airports
    )
    assert len(flight_pieces.km) == 2
    assert code_to_id["CDG"] in (0, 1, 2)
    assert steps[1]["routes"] == 2


def test_universal_intermodal_hub_anchoring(tmp_path):
    """Verify that generic IntermodalHub anchors multiple modes with medium checks."""
    # Build road network: Line between (48.85, 2.35) and (48.86, 2.36)
    road_pieces = assembly.PieceArrays(tolerance_meters=None)
    road_pieces.append(0, 1, 5.0, 0, [[48.85, 2.35], [48.86, 2.36]], [])
    road_graph = assembly.build_graph_from_pieces(road_pieces)

    # Build rail network: Line between (48.84, 2.34) and (48.87, 2.37)
    rail_pieces = assembly.PieceArrays(tolerance_meters=None)
    rail_pieces.append(0, 1, 8.0, 0, [[48.84, 2.34], [48.87, 2.37]], [])
    rail_graph = assembly.build_graph_from_pieces(rail_pieces)

    # Land mask where everything is dry land
    width, height = 100, 100
    stride = (width + 7) // 8
    bits = bytearray([0xFF] * (height * stride))
    mask = LandMask(
        bits=bits,
        south=48.0,
        west=2.0,
        height=height,
        width=width,
        cells_per_degree=100,
    )

    # Intermodal hub connecting Road + Rail (Gare de Lyon / Dry Port)
    hub = IntermodalHub(
        latitude=48.845,
        longitude=2.345,  # Close to road and rail
        hub_type="rail_terminal",
        capabilities=["road", "rail"],
    )

    modal_graphs = {"road": road_graph, "rail": rail_graph}
    joined, counts = hubs.join_multimodal_hubs(
        modal_graphs,
        [hub],
        mask,
        tolerances={"road": 50.0, "rail": 50.0},
    )
    assert counts["hubs"] == 1
    assert "node_modes" in joined
    assert len(joined["node_modes"]) == len(joined["node_ids"])


def test_pareto_shortest_paths_monotonicity():
    """Verify duration vs CO2 Pareto monotonicity and modal transition."""
    # Build a small multimodal corridor:
    # Origin -> (road) -> Hub_Airport -> (air) -> Hub_Airport_Dest -> (road) -> Destination
    # Origin -> (road) -> Hub_Rail -> (rail) -> Hub_Rail_Dest -> (road) -> Destination
    nodes = [
        [0.0, 0.0],  # 0: Origin
        [0.0, 0.1],  # 1: Road junction
        [0.0, 0.2],  # 2: Air Hub Origin
        [10.0, 0.2],  # 3: Air Hub Dest
        [0.0, 0.3],  # 4: Rail Hub Origin
        [10.0, 0.3],  # 5: Rail Hub Dest
        [10.0, 0.1],  # 6: Road junction Dest
        [10.0, 0.0],  # 7: Destination
    ]
    # Modes: 0: road (0,1,6,7), 1: air (2,3), 2: rail (4,5), 3: hub
    modes_meta = [
        {"name": "road"},
        {"name": "air"},
        {"name": "rail"},
        {"name": "hub"},
    ]
    node_modes = [0, 0, 1, 1, 2, 2, 0, 0]

    # Graph distances:
    # Road access
    adj = {
        0: {1: 10.0},
        1: {0: 10.0, 2: 10.0, 4: 10.0},
        # Transfers
        2: {1: 10.0, 3: 1000.0},  # Air flight 1000 km
        3: {2: 1000.0, 6: 10.0},
        4: {1: 10.0, 5: 1100.0},  # Rail corridor 1100 km
        5: {4: 1100.0, 6: 10.0},
        6: {3: 10.0, 5: 10.0, 7: 10.0},
        7: {6: 10.0},
    }
    graph_list = [adj[i] for i in range(len(nodes))]

    geo = GeoGraph(
        graph=graph_list,
        nodes=nodes,
        modes=modes_meta,
        node_modes=node_modes,
    )

    pareto_routes = geo.get_pareto_shortest_paths(
        origin_node={"latitude": 0.0, "longitude": 0.0},
        destination_node={"latitude": 10.0, "longitude": 0.0},
        lambdas=[0.0, 0.01, 1.0, 10.0],
        auto_lat_lon_bound_max=15.0,
    )

    assert len(pareto_routes) >= 2
    # Check monotonicity: Duration non-decreasing, CO2 non-increasing
    for i in range(len(pareto_routes) - 1):
        r1, r2 = pareto_routes[i], pareto_routes[i + 1]
        assert r1["duration_hours"] <= r2["duration_hours"]
        assert r1["co2_kg"] >= r2["co2_kg"]

    # At lambda = 0 (speed priority), Air should be used
    fastest = pareto_routes[0]
    assert "air" in fastest["modes"] or "hub" in fastest["modes"]


def test_hub_admissibility_drops_isolated_hubs():
    """Verify that hubs connecting to fewer than 2 modes are dropped."""
    # Road network only
    road_pieces = assembly.PieceArrays(tolerance_meters=None)
    road_pieces.append(0, 1, 5.0, 0, [[48.85, 2.35], [48.86, 2.36]], [])
    road_graph = assembly.build_graph_from_pieces(road_pieces)

    # Empty water mask
    width, height = 50, 50
    stride = (width + 7) // 8
    mask = LandMask(
        bits=bytearray([0xFF] * (height * stride)),
        south=48.0,
        west=2.0,
        height=height,
        width=width,
        cells_per_degree=100,
    )

    # Hub that only declares rail, but rail is not in modal_graphs
    isolated_hub = IntermodalHub(
        latitude=48.85,
        longitude=2.35,
        hub_type="rail_terminal",
        capabilities=["rail"],
    )

    joined, counts = hubs.join_multimodal_hubs(
        {"road": road_graph},
        [isolated_hub],
        mask,
    )
    # Must be dropped because it only connected to 0 modes
    assert counts["hubs"] == 0


def test_world_multimodal_pipeline_metadata():
    """Verify the world_multimodal pipeline contract and declaration."""
    from scgraph.pipelines import world_multimodal

    assert world_multimodal.PIPELINE_VERSION == 1
    assert "roads" in world_multimodal.SOURCES
    assert "sea" in world_multimodal.SOURCES
    assert "rail" in world_multimodal.SOURCES
    assert "airports" in world_multimodal.SOURCES
    assert "flights" in world_multimodal.SOURCES
    assert "ports" in world_multimodal.SOURCES

    mode_names = [m["name"] for m in world_multimodal.MODES]
    assert mode_names == ["road", "sea", "rail", "air", "hub"]
    assert "cases" in world_multimodal.VALIDATION
