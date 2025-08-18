from geodoc_config import load_config, load_config_by_path

COLLECTIONS_CONFIG = load_config('app_collections')

def get_collections_list():
    """
    Get list of collections from config.
    Returns:
        list: List of collection display names.
    """
    return {
        "title": COLLECTIONS_CONFIG['title'],
        "default": COLLECTIONS_CONFIG['default'],
        "items": [v['name'] for v in COLLECTIONS_CONFIG['collections'].values()],
        "children": COLLECTIONS_CONFIG.get('children', False),
        "selector_type": "combo_box",
        "symbols": ['collections']
    }

def get_collection_item_config(symbol):
    """
    Get collection item config by symbol.
    Args:
        symbol (str): Symbol of the collection item.
    Returns:
        dict: Collection item configuration.
    """
    item_spec = COLLECTIONS_CONFIG['collections'].get(symbol, None)
    if item_spec is None:
        raise ValueError(f"Collection with symbol '{symbol}' not found in config.")
    return load_config_by_path(item_spec['folder'], item_spec['file'])

def get_collection_symbol_by_name(name):
    """
    Get collection symbol by display name.
    Args:
        name (str): Display name of the collection.
    Returns:
        str: Symbol of the collection.
    """
    for symbol, item in COLLECTIONS_CONFIG['collections'].items():
        if item['name'] == name:
            return symbol
    raise ValueError(f"Collection with display name '{name}' not found in config.")

def get_collection_tables(collection_config):
    """
    Get list of tables from collection config.
    Args:
        collection_config (dict): Collection configuration.
    Returns:
        list: List of table names in the collection.
    """
    return [table['name'] for table in collection_config['tables'].values()]

def get_collection_tables_by_name(name, symbols=['collections']):
    """
    Get list of tables by collection name.
    Args:
        name (str): Display name of the collection.
    Returns:
        list: List of table names in the collection.
    """
    symbol = get_collection_symbol_by_name(name)
    if symbol is None:
        return None
    collection_config = get_collection_item_config(symbol)
    return {
        "title": collection_config['title'],
        "default": collection_config.get('default', ''),
        "items": get_collection_tables(collection_config),
        "children": collection_config.get('children', False),
        "selector_type": "combo_box",
        "symbols": symbols + [symbol]
    }

def get_collection_table_spec_by_name(collection_symbol, table_name):
    """
    Get table specification by collection symbol and table name.
    Args:
        collection_symbol (str): Symbol of the collection.
        table_name (str): Name of the table.
    Returns:
        dict: Table specification.
    """
    collection_config = get_collection_item_config(collection_symbol)
    for table in collection_config['tables'].values():
        if table['name'] == table_name:
            return table
    raise ValueError(f"Table '{table_name}' not found in collection '{collection_symbol}'.")

def get_collection_table_columns_spec_by_name(collection_symbol, table_name):
    """
    Get columns specification for a table in a collection.
    Args:
        collection_symbol (str): Symbol of the collection.
        table_name (str): Name of the table.
    Returns:
        list: List of column specifications.
    """
    table_spec = get_collection_table_spec_by_name(collection_symbol, table_name)
    return table_spec.get('columns', [])

def get_filter_spec(symbols, name=None):
    print(f'symbols: {symbols} | name: {name}')
    if symbols is None:
        return None
    elif len(symbols) == 0:
        return get_collections_list()
    elif len(symbols) == 1 and symbols[0] == 'collections':
        return get_collection_tables_by_name(name=name, symbols=symbols)
    
    return None