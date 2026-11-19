"""Tests of `scgraph.pipelines.derive`: a network file derived from a parent file."""

import pytest

from scgraph import GeoGraph, pipelines
from scgraph.pipelines import derive

# The parent network of these tests. A place is `id: [latitude, longitude]`.
# A road joins the places 1, 2 and 3 along the equator. A second road goes
# from place 2 to place 4, two degrees north. The places 5 and 6 are the road
# of an island. Place 10 is a port. It joins place 3 to the sea lane from
# place 20 to place 21, which is at 50 degrees east.
ROAD_PLACES = {1: [0, 0], 2: [0, 0.03], 3: [0, 0.06], 4: [2, 0.03]}
ISLAND_PLACES = {5: [0, 5], 6: [0, 5.03]}
PORT_PLACES = {10: [0, 0.07]}
SEA_PLACES = {20: [-0.1, 0.07], 21: [-0.1, 50]}
PLACES = ROAD_PLACES | ISLAND_PLACES | PORT_PLACES | SEA_PLACES
# Each link is (place, place, km), and gives one arc each way.
ROAD_LINKS = [(1, 2, 3.0), (2, 3, 3.0), (2, 4, 222.0), (5, 6, 3.0)]
PORT_AND_SEA_LINKS = [(3, 10, 1.0), (10, 20, 11.0), (20, 21, 5550.0)]
# The road from place 1 to place 2 bends at this point: the only arc shape.
BEND = [0.01, 0.015]
MODES = [
    {"name": "road", "source": "osm"},
    {"name": "port", "source": "ports"},
    {"name": "sea", "source": "sea"},
]
# No mode cites the source `land`, as in a multimodal pipeline.
SOURCES = [{"name": name} for name in ("osm", "ports", "sea", "land")]
# The rectangle (latitude min, latitude max, longitude min, longitude max)
# holds the places 1, 2 and 3. Place 4, the island and place 21 are outside.
RECTANGLE = (-1, 1, -1, 1)


@pytest.fixture(scope="module")
def parent_file(tmp_path_factory):
    """The network file of the parent network, with the header that a
    pipeline writes."""
    node = {place: u for u, place in enumerate(PLACES)}
    graph, shapes = [{} for _ in PLACES], [{} for _ in PLACES]
    for a, b, km in ROAD_LINKS + PORT_AND_SEA_LINKS:
        graph[node[a]][node[b]] = graph[node[b]][node[a]] = km
    shapes[node[1]][node[2]] = shapes[node[2]][node[1]] = [BEND]
    path = str(tmp_path_factory.mktemp("parent") / "parent.scgraph")
    GeoGraph(
        graph=graph,
        nodes=list(PLACES.values()),
        intermediate_nodes=shapes,
        node_ids=list(PLACES),
        node_modes=[0] * 6 + [1] + [2] * 2,  # rows of MODES, as PLACES
        modes=MODES,
        coordinate_dtype="int32",
    ).save_as_scgraph(
        path,
        metadata=pipelines.build_metadata("world_multimodal", None, 1, SOURCES),
    )
    return path


# The networks of these tests are small, so each derivation keeps a component
# of any size. One test below gives another smallest size.
@pytest.fixture(scope="module")
def sea_file(parent_file, tmp_path_factory):
    """The file derived from the parent file with the modes of the sea."""
    path = str(tmp_path_factory.mktemp("sea") / "sea.scgraph")
    derive.derive_file(
        parent_file, path, modes=["sea", "port"], min_component_nodes=1
    )
    return path


@pytest.fixture(scope="module")
def region_file(parent_file, tmp_path_factory):
    """The file derived from the parent file with the rectangle."""
    path = str(tmp_path_factory.mktemp("region") / "region.scgraph")
    derive.derive_file(
        parent_file, path, rectangle=RECTANGLE, min_component_nodes=1
    )
    return path


def _places_by_mode(file):
    """The places of a network file: for each mode, the sorted ids of its nodes."""
    network = GeoGraph.load_from_scgraph(file)
    places = {mode["name"]: [] for mode in network.modes}
    for row, place in zip(network.node_modes, network.node_ids):
        places[network.modes[row]["name"]].append(place)
    return {mode: sorted(ids) for mode, ids in places.items()}


def _km_from_place_1_to_place_21(file):
    """The length of the shortest route of a network file from the west end
    of the road to the east end of the sea lane."""
    return GeoGraph.load_from_scgraph(file).get_shortest_path(
        {"latitude": 0, "longitude": 0}, {"latitude": -0.1, "longitude": 50}
    )["length"]


def test_a_derived_file_holds_only_the_nodes_of_the_modes_asked(sea_file):
    assert _places_by_mode(sea_file) == {"port": [10], "sea": [20, 21]}


def test_a_derived_file_cites_only_the_sources_that_serve_its_modes(sea_file):
    sources = GeoGraph.get_scgraph_info(sea_file)["sources"]

    # Only the mode `road`, which is removed, cites `osm`.
    assert sources == [{"name": "ports"}, {"name": "sea"}, {"name": "land"}]


def test_a_rectangle_cuts_the_roads_and_keeps_every_sea_node(region_file):
    # Place 2 is inside the rectangle, and its road to place 4 is cut. It is
    # then between the places 1 and 3 only, so it is contracted.
    assert _places_by_mode(region_file) == {
        "road": [1, 3],
        "port": [10],
        "sea": [20, 21],
    }


def test_a_route_of_a_derived_file_has_its_length_in_the_parent_file(
    parent_file, region_file
):
    # In both files: 3 + 3 km of road, 1 + 11 km at the port, 5550 km of sea
    assert _km_from_place_1_to_place_21(region_file) == pytest.approx(
        _km_from_place_1_to_place_21(parent_file)
    )


def test_a_file_without_shapes_holds_the_network_of_the_file_with_shapes(
    parent_file, region_file, tmp_path
):
    bare_file = str(tmp_path / "region.no_shapes.scgraph")

    derive.derive_file(
        parent_file,
        bare_file,
        rectangle=RECTANGLE,
        keep_shapes=False,
        min_component_nodes=1,
    )

    assert GeoGraph.get_scgraph_info(bare_file)["shapes"] is False
    assert _km_from_place_1_to_place_21(
        bare_file
    ) == _km_from_place_1_to_place_21(region_file)


def test_a_component_is_kept_from_the_smallest_size_asked(
    parent_file, tmp_path
):
    roads_file = str(tmp_path / "roads.scgraph")

    derive.derive_file(
        parent_file, roads_file, modes=["road"], min_component_nodes=4
    )

    # The road of the places 1 to 4 has four nodes, and the island has two.
    assert _places_by_mode(roads_file) == {"road": [1, 2, 3, 4]}


def test_the_header_of_a_derived_file_records_its_parent_and_its_arguments(
    parent_file, region_file
):
    derivation = GeoGraph.get_scgraph_info(region_file)["derivation"]

    assert derivation == {
        "parent_content_sha256": GeoGraph.get_scgraph_info(parent_file)[
            "content_sha256"
        ],
        "modes": None,
        "rectangle": list(RECTANGLE),
        "min_component_nodes": 1,
    }
