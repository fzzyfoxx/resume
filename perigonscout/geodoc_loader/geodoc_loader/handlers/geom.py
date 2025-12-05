from shapely import wkt
import shapely.ops as ops
import pyproj

def convert_geom_from_wkt(wkt_geom, in_crs='EPSG:4326', out_crs='EPSG:2180'):
    """
    Convert WKT geometry from one CRS to another.
    
    Args:
        wkt_geom (str): WKT representation of the geometry.
        in_crs (str): Input CRS in EPSG format (default is 'EPSG:4326').
        out_crs (str): Output CRS in EPSG format (default is 'EPSG:2180').
    Returns:
        shapely.geometry: Geometry object in the output CRS.
    """
    in_crs = pyproj.CRS(in_crs)
    out_crs = pyproj.CRS(out_crs)
    shp_geom = wkt.loads(wkt_geom)
    
    project = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True).transform
    out_geom = ops.transform(project, shp_geom)
    return out_geom

def convert_geoms_from_wkt(wkt_geoms, in_crs='EPSG:4326', out_crs='EPSG:2180', geom_key='geometry'):
    """
    Convert a list of WKT geometries from one CRS to another.
    
    Args:
        wkt_geoms (list): List of dictionaries where wkt geometry is under geom_key.
        in_crs (str): Input CRS in EPSG format (default is 'EPSG:4326').
        out_crs (str): Output CRS in EPSG format (default is 'EPSG:2180').
        geom_key (str): Key in the dictionary where the WKT geometry is stored (default is 'geometry').
    Returns:
        list: List of dictionaries with geom_key replaced by shapely.geometry in the output CRS.
    """

    converted_geoms = []
    for item in wkt_geoms:
        if geom_key in item:
            item[geom_key] = convert_geom_from_wkt(item[geom_key], in_crs, out_crs)
        converted_geoms.append(item)
    return converted_geoms

def get_bbox_bottom_left(geom):
    """
    Get bottom left corner of the shapely.geometry.

    Args:
        geom (shapely.geometry): Input geometry.
    Returns:
        tuple: (x, y) coordinates of the bottom left corner.
    """
    # get bottom left corner
    x = geom.bounds[0]
    y = geom.bounds[1]
    
    return x, y

def transform_coordinates(x, y, source_crs, target_crs):
    """
    Transforms coordinates from source CRS to target CRS.

    Args:
        x (float): X coordinate in the source CRS.
        y (float): Y coordinate in the source CRS.
        source_crs (str): Source CRS in EPSG format (e.g., 'EPSG:4326').
        target_crs (str): Target CRS in EPSG format (e.g., 'EPSG:3857').

    Returns:
        tuple: Transformed coordinates (x_new, y_new) in the target CRS.
    """
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x_new, y_new = transformer.transform(x, y)
    return x_new, y_new
    


