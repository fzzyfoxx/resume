import json
import csv
import io
from shapely.geometry import shape as shapely_shape
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Polygon, MultiPolygon
from pyproj import Geod
from ast import literal_eval

GEOD = Geod(ellps="WGS84")

def wkt_from_geojson(geometry):
    if geometry:
        return shapely_shape(geometry).wkt
    return None

def _polygon_area_m2(poly: Polygon):
    exterior_lon, exterior_lat = zip(*poly.exterior.coords)
    area, _ = GEOD.polygon_area_perimeter(exterior_lon, exterior_lat)
    total = abs(area)
    for interior in poly.interiors:
        hole_lon, hole_lat = zip(*interior.coords)
        h_area, _ = GEOD.polygon_area_perimeter(hole_lon, hole_lat)
        total -= abs(h_area)
    return total

def _geodetic_area_m2(geom):
    if geom.is_empty:
        return 0.0
    if isinstance(geom, Polygon):
        return _polygon_area_m2(geom)
    if isinstance(geom, MultiPolygon):
        return sum(_polygon_area_m2(p) for p in geom.geoms)
    return 0.0

def _safe_round(val):
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return None

def _as_list(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    s = value.strip()
    if not s:
        return []
    # Try JSON (valid JSON must have quoted strings)
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # Try Python literal list form
    try:
        parsed = literal_eval(s)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # Fallback: strip outer brackets then split on commas not inside parentheses
    if s.startswith('[') and s.endswith(']'):
        s_inside = s[1:-1].strip()
    else:
        s_inside = s
    if not s_inside:
        return []
    parts = []
    current = []
    depth = 0
    for ch in s_inside:
        if ch == ',' and depth == 0:
            token = ''.join(current).strip()
            if token:
                parts.append(token)
            current = []
            continue
        current.append(ch)
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = depth - 1 if depth > 0 else 0
    tail = ''.join(current).strip()
    if tail:
        parts.append(tail)
    cleaned = [p.strip().strip('"').strip("'") for p in parts if p.strip()]
    return cleaned

def extract_filter_features(properties, geometry):

    data = {
        'typ': 'filtr',
        'nazwa': properties.get('name', 'Brak nazwy'),
        'kwalifikacja': properties.get('Kwalifikacja'),
        'bufor': properties.get('bufor'),
        'geometria': wkt_from_geojson(geometry)
    }

    columns = list(data.keys())

    return [data], columns

def extract_search_features(properties, geometry):

    data = {
        'typ': 'obszar wyszukiwania',
        'geometria': wkt_from_geojson(geometry)
    }

    columns = list(data.keys())

    return [data], columns

def extract_parcel_features(properties, geometry):

    parcels_geometries = _as_list(properties.get('geometry', None))
    parcels_ids = _as_list(properties.get('parcel_id', None))

    area_val = _safe_round(properties.get('area', None))
    qual_val = _safe_round(properties.get('qualified_area', None))

    data = {
        'typ': 'znaleziony obszar',
        'identyfikatory działek': parcels_ids if parcels_ids else None,
        'powierzchnia': area_val,
        'powierzchnia zakwalifikowana': qual_val,
        'liczba działek': properties.get('parcels_num', None),
        'geometria': wkt_from_geojson(geometry),
    }

    rows = [data]
    columns = list(data.keys())

    if parcels_geometries and parcels_ids and geometry:
        general_geom = shapely_shape(geometry)
        for wkt_str, parcel_id in zip(parcels_geometries, parcels_ids):
            try:
                parcel_geom = wkt_loads(wkt_str)
            except Exception:
                continue
            parcel_area = _geodetic_area_m2(parcel_geom)
            inter = parcel_geom.intersection(general_geom)
            qualified_area = _geodetic_area_m2(inter) if not inter.is_empty else 0.0
            row = {
                'typ': 'działka',
                'identyfikatory działek': parcel_id,
                'powierzchnia': round(parcel_area, 2),
                'powierzchnia zakwalifikowana': round(qualified_area, 2),
                'liczba działek': None,
                'geometria': wkt_str
            }
            rows.append(row)

    return rows, columns

def extract_features(features, geometry):

    properties = features.get('properties', {})

    result_type = properties.get('type', None)
    filename_base = properties.get('name') or properties.get('id') or 'feature'

    if result_type == 'FilterResult':
        rows, columns = extract_filter_features(properties, geometry)
    elif result_type == 'SearchArea':
        rows, columns = extract_search_features(properties, geometry)
    elif result_type == 'ParcelTarget':
        rows, columns = extract_parcel_features(properties, geometry)
    else:
        rows, columns =  [], []

    return rows, columns, filename_base

