"""Extract railway networks from open sources into PieceArrays."""

import json
import logging
import os
from collections import Counter

from scgraph.helpers.polylines import cut_polylines
from scgraph.pipelines.steps.stage3_assembly.assembly import PieceArrays

logger = logging.getLogger(__name__)

NATURAL_EARTH_RAIL = {
    "url": (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "master/geojson/ne_10m_railroads.geojson"
    ),
    "credit": "Natural Earth (1:10m Railroads)",
    "version": "5.1.2",
    "license": "CC0-1.0",
}


def extract_rail_pieces(
    filename: str | os.PathLike,
    *,
    source_type: str = "natural_earth",
    max_km_between_nodes: float = 10.0,
    tolerance_meters: float = 50.0,
    work_dir: str | os.PathLike | None = None,
) -> tuple[PieceArrays, list[dict[str, str | int]]]:
    """Extract railway arcs into PieceArrays.

    Args:
        filename: Path to the input railway dataset.
        source_type: Format and extractor type ('natural_earth' or 'osm').
        max_km_between_nodes: Maximum spacing in kilometers between graph nodes.
        tolerance_meters: Douglas-Peucker shape simplification tolerance.
        work_dir: Optional directory for intermediate disk-backed arrays.

    Returns:
        Tuple of (PieceArrays, list of step metrics).
    """
    if source_type == "osm":
        from scgraph.pipelines.steps.stage1_extract import osm

        return osm.build_way_pieces(
            osm.open_osm_source(filename),
            {
                "rail": (
                    osm.select_rail
                    if hasattr(osm, "select_rail")
                    else _select_rail
                )
            },
            ("railway",),
            max_km_between_nodes=max_km_between_nodes,
            work_dir=work_dir,
        )

    if source_type != "natural_earth":
        raise ValueError(f"Unsupported railway source type: {source_type}")

    return _extract_natural_earth_rail(
        filename,
        max_km_between_nodes=max_km_between_nodes,
        tolerance_meters=tolerance_meters,
        work_dir=work_dir,
    )


def _select_rail(tags) -> int | None:
    """OSM railway tag selector (both directions)."""
    return 0 if tags.get("railway") == "rail" else None


def _extract_natural_earth_rail(
    filename: str | os.PathLike,
    *,
    max_km_between_nodes: float,
    tolerance_meters: float,
    work_dir: str | os.PathLike | None,
) -> tuple[PieceArrays, list[dict[str, str | int]]]:
    """Parse Natural Earth 10m railroads GeoJSON into PieceArrays."""
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    raw_lines = []
    coord_counts = Counter()
    coord_to_id = {}

    def get_node_id(lon: float, lat: float) -> int:
        key = (round(lat, 7), round(lon, 7))
        if key not in coord_to_id:
            coord_to_id[key] = len(coord_to_id)
        return coord_to_id[key]

    for feat in features:
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            continue
        lines = (
            [coords]
            if gtype == "LineString"
            else coords if gtype == "MultiLineString" else []
        )
        for line in lines:
            if len(line) < 2:
                continue
            # Natural Earth coords are [longitude, latitude]
            pts = [(get_node_id(pt[0], pt[1]), [pt[1], pt[0]]) for pt in line]
            raw_lines.append(pts)
            coord_counts[pts[0][0]] += 2  # Endpoints act as junctions
            coord_counts[pts[-1][0]] += 2
            for nid, _ in pts[1:-1]:
                coord_counts[nid] += 1

    junctions = {nid for nid, count in coord_counts.items() if count >= 2}
    polylines = [(0, pts) for pts in raw_lines]  # 0 = bidirectional rail

    pieces = PieceArrays(
        tolerance_meters=tolerance_meters,
        work_dir=work_dir,
    )
    where = {nid: pt for pts in raw_lines for nid, pt in pts}

    for origin, destination, km, direction, shape in cut_polylines(
        polylines, junctions, max_km_between_nodes
    ):
        ends = [where[origin], where[destination]]
        pieces.append(origin, destination, km, direction, ends, shape)

    pieces.flush()
    steps = [
        {"step": "read_rail", "lines": len(raw_lines), "pieces": len(pieces.km)}
    ]
    return pieces, steps
