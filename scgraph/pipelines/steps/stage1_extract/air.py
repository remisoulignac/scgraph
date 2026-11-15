"""Extract airports and commercial flight networks into PieceArrays."""

import csv
import logging
import os
from collections.abc import Sequence

from scgraph.pipelines.steps.stage3_assembly.assembly import PieceArrays
from scgraph.utils import haversine

logger = logging.getLogger(__name__)

OURAIRPORTS = {
    "url": "https://davidmegginson.github.io/ourairports-data/airports.csv",
    "credit": "OurAirports",
    "version": "2026",
    "license": "CC0-1.0",
}

OPENFLIGHTS_ROUTES = {
    "url": (
        "https://raw.githubusercontent.com/jpatokal/openflights/master/data/"
        "routes.dat"
    ),
    "credit": "OpenFlights",
    "version": "2026",
    "license": "ODbL-1.0",
}


def read_airports(
    filename: str | os.PathLike,
    *,
    min_types: Sequence[str] = ("large_airport", "medium_airport"),
    scheduled_only: bool = True,
) -> dict[str, dict[str, str | float]]:
    """Parse OurAirports CSV dataset.

    Args:
        filename: Path to OurAirports airports.csv.
        min_types: Airport classification tiers to retain.
        scheduled_only: If True, only keep airports with commercial service.

    Returns:
        Mapping of airport identifier (IATA or ident) to airport metadata.
    """
    airports = {}
    with open(filename, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            atype = row.get("type", "")
            scheduled = row.get("scheduled_service", "").lower()
            if atype not in min_types:
                continue
            if scheduled_only and scheduled != "yes":
                continue
            iata = row.get("iata_code", "").strip()
            ident = row.get("ident", "").strip()
            key = iata if iata else ident
            if not key:
                continue
            try:
                lat = float(row["latitude_deg"])
                lon = float(row["longitude_deg"])
            except (ValueError, KeyError):
                continue
            airports[key] = {
                "ident": ident,
                "iata": iata,
                "name": row.get("name", ""),
                "type": atype,
                "latitude": lat,
                "longitude": lon,
            }
    return airports


def extract_flight_pieces(
    routes_filename: str | os.PathLike,
    airports: dict[str, dict[str, str | float]],
    *,
    work_dir: str | os.PathLike | None = None,
) -> tuple[PieceArrays, dict[str, int], list[dict[str, str | int]]]:
    """Parse OpenFlights route database and create flight arcs in PieceArrays.

    Args:
        routes_filename: Path to OpenFlights routes.dat CSV.
        airports: Mapping of airport keys to airport records from read_airports.
        work_dir: Optional intermediate storage directory for PieceArrays.

    Returns:
        Tuple of (PieceArrays, mapping of airport code to node id, steps).
    """
    code_to_id = {code: i for i, code in enumerate(sorted(airports.keys()))}
    pieces = PieceArrays(tolerance_meters=None, work_dir=work_dir)
    seen_arcs = set()

    with open(routes_filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            src, dst = row[2].strip(), row[4].strip()
            if src not in code_to_id or dst not in code_to_id:
                continue
            if src == dst:
                continue
            u, v = code_to_id[src], code_to_id[dst]
            if (u, v) in seen_arcs:
                continue
            seen_arcs.add((u, v))

            src_pt = [airports[src]["latitude"], airports[src]["longitude"]]
            dst_pt = [airports[dst]["latitude"], airports[dst]["longitude"]]
            dist_km = haversine(src_pt, dst_pt)

            # Direction 1 = directed flight arc
            pieces.append(u, v, dist_km, 1, [src_pt, dst_pt], [])

    pieces.flush()
    steps = [
        {"step": "airports", "count": len(airports)},
        {"step": "flights", "routes": len(pieces.km)},
    ]
    return pieces, code_to_id, steps
