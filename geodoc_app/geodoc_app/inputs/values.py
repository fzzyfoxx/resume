from geodoc_config import load_config, load_config_by_path

COLLECTIONS_CONFIG = load_config('app_collections')

def get_collections_list():
    """
    Get list of collections from config.
    Returns:
        list: List of collection display names.
    """
    answer = {
        "title": COLLECTIONS_CONFIG['title'],
        "default": COLLECTIONS_CONFIG['default'],
        "items": [v['name'] for v in COLLECTIONS_CONFIG['collections'].values()],
        "children": COLLECTIONS_CONFIG.get('children', False),
        "selector_type": "combo_box",
        "symbols": ['collections']
    }

    return {'filters': [answer]}

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
    answer = {
        "title": collection_config['title'],
        "default": collection_config.get('default', ''),
        "items": get_collection_tables(collection_config),
        "children": collection_config.get('children', False),
        "selector_type": "combo_box",
        "symbols": symbols + [symbol]
    }

    return {"filters": [answer]}

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
    for table_symbol, table in collection_config['tables'].items():
        if table['name'] == table_name:
            return {**table, 'table_symbol': table_symbol}
    return None

def get_selector_type(column_type):
    """
    Get selector type based on column type.
    Args:
        column_type (str): Type of the column.
    Returns:
        str: Selector type.
    """
    if column_type == 'CATEGORICAL':
        return 'select'
    elif column_type == 'SEARCH':
        return 'search'
    elif column_type == 'NUMERIC':
        return 'numeric'
    
def get_items_for_column(column_spec, column_symbol, table_symbol, collection_symbol, mappings=None):

    column_type = column_spec['type']
    if column_type == 'CATEGORICAL':
        try:
            items =  load_config_by_path(f'search.{collection_symbol}.columns', f'{table_symbol}_{column_symbol}.json')['values']
            mapping_code = column_spec.get('mapping', None)

            if mapping_code is not None and mappings is not None:
                reversed_mapping = {v: k for k, v in mappings[mapping_code].items()}
                items = [reversed_mapping.get(item, item) for item in items]

            null_value = column_spec.get('ifnull', None)
            if null_value is not None:
                items.append(null_value)
        except Exception as e:
            print(e)
            items = []
        
        return items
    
    return None

def get_column_items_from_symbols(symbols):
    try:
        _, collection_symbol, table_symbol, column_symbol = symbols
        return load_config_by_path(f'search.{collection_symbol}.columns', f'{table_symbol}_{column_symbol}.json')
    except Exception as e:
        return []
    

def get_collection_table_columns_spec_by_name(name, symbols):
    """
    Get columns specification for a table in a collection.
    Args:
        collection_symbol (str): Symbol of the collection.
        table_name (str): Name of the table.
    Returns:
        list: List of column specifications.
    """
    table_spec = get_collection_table_spec_by_name(symbols[-1], name)
    columns_def = table_spec.get('columns', {})
    mappings = table_spec.get('mappings', None)
    answer = []
    for i, (column, column_spec) in enumerate(columns_def.items()):
        answer.append({
            "title": column_spec['name'],
            "default": column_spec.get('default', ''),
            "items": get_items_for_column(column_spec, column, table_spec['table_symbol'], symbols[-1], mappings),
            "children": column_spec.get('children', False),
            "selector_type": get_selector_type(column_spec['type']),
            "symbols": symbols + [table_spec['table_symbol'], column]
        })
    
    return {'filters': answer}

def get_filter_spec(symbols, name=None):
    print(f'symbols: {symbols} | name: {name}')
    if symbols is None:
        return None
    
    if len(symbols) == 0:
        answer = get_collections_list()
        answer_item = answer['filters'][0]
        if len(answer_item['items']) == 1:
            symbols = answer_item['symbols']
            name = answer_item['items'][0]
        else:
            return answer
    
    if len(symbols) == 1 and symbols[0] == 'collections':
        answer = get_collection_tables_by_name(name=name, symbols=symbols)
        answer_item = answer['filters'][0]
        if len(answer_item['items']) == 1:
            symbols = answer_item['symbols']
            name = answer_item['items'][0]
        else:
            return answer
        
    if len(symbols) == 2 and symbols[0] == 'collections':
        answer = get_collection_table_columns_spec_by_name(name=name, symbols=symbols)
        return answer

    return None