from geodoc_config import load_config, load_config_by_path

COLLECTIONS_CONFIG = load_config('app_collections')

def get_subtitle_config(title):
    """
    Get subtitle configuration.
    Args:
        title (str): Title of the subtitle.
    Returns:
        dict: Subtitle configuration.
    """
    return {
        "title": title,
        "default": title,
        "items": [],
        "children": False,
        "selector_type": "subtitle",
        "symbols": ['subtitle', title],
        "ispassive": True,
    }

def get_qualification_config():
    """
    Get qualification configuration.
    Returns:
        dict: Qualification configuration.
    """
    qualification_spec = load_config_by_path('search.utils', 'qualification.json')
    answer = {
        "title": qualification_spec['title'],
        "default": qualification_spec['default'],
        "items": qualification_spec['options'],
        "children": False,
        "selector_type": "qualification",
        "symbols": ['qualification'],
        "ispassive": False
    }

    return answer

def get_collections_list(collection_name='collections'):
    """
    Get list of collections from config.
    Args:
        collection_name (str): Name of the collection.
    Returns:
        list: List of collection display names.
    """
    collection_config = COLLECTIONS_CONFIG[collection_name]
    answer = {
        "title": collection_config['title'],
        "default": collection_config['default'],
        "items": [v['name'] for v in collection_config['collections'].values()],
        "children": collection_config.get('children', False),
        "selector_type": "combo_box",
        "symbols": [collection_name],
        "ispassive": False
    }

    return {'filters': [answer]}

def get_collection_item_config(collection_name, symbol):
    """
    Get collection item config by symbol.
    Args:
        collection_name (str): Name of the collection.
        symbol (str): Symbol of the collection item.
    Returns:
        dict: Collection item configuration.
    """
    item_spec = COLLECTIONS_CONFIG[collection_name]['collections'].get(symbol, None)
    if item_spec is None:
        raise ValueError(f"Collection with symbol '{symbol}' not found in config.")
    return load_config_by_path(item_spec['folder'], item_spec['file'])

def get_collection_symbol_by_name(collection_name, name):
    """
    Get collection symbol by display name.
    Args:
        collection_name (str): Name of the collection.
        name (str): Display name of the collection.
    Returns:
        str: Symbol of the collection.
    """
    for symbol, item in COLLECTIONS_CONFIG[collection_name]['collections'].items():
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

def get_collection_tables_by_name(collection_name, name, symbols=['collections']):
    """
    Get list of tables by collection name.
    Args:
        collection_name (str): Name of the collection.
        name (str): Display name of the collection.
    Returns:
        list: List of table names in the collection.
    """
    symbol = get_collection_symbol_by_name(collection_name, name)
    if symbol is None:
        return None
    collection_config = get_collection_item_config(collection_name, symbol)
    answer = {
        "title": collection_config['title'],
        "default": collection_config.get('default', ''),
        "items": get_collection_tables(collection_config),
        "children": collection_config.get('children', False),
        "selector_type": "combo_box",
        "symbols": symbols + [symbol],
        "ispassive": False
    }

    return {"filters": [answer]}

def get_collection_table_spec_by_name(collection_name, collection_symbol, table_name):
    """
    Get table specification by collection symbol and table name.
    Args:
        collection_symbol (str): Symbol of the collection.
        table_name (str): Name of the table.
    Returns:
        dict: Table specification.
    """
    collection_config = get_collection_item_config(collection_name, collection_symbol)
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
    elif column_type == 'DICTSEARCH':
        return 'dict_search'
    elif column_type == 'UNITNUMERIC':
        return 'unit_numeric'
    elif column_type == 'SWITCH':
        return 'switch'
    elif column_type == 'EQUALSNUMERIC':
        return 'equals_numeric'
    
def get_items_for_column(column_spec, column_symbol, table_symbol, collection_symbol, mappings=None):

    column_type = column_spec['type']
    if column_type in ['CATEGORICAL', 'UNITNUMERIC']:
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
    elif column_type == 'DICTSEARCH':
        items = column_spec.get('display_keys', [])
        return items
    return None

def get_column_items_from_symbols(symbols):
    try:
        _, collection_symbol, table_symbol, column_symbol = symbols
        return load_config_by_path(f'search.{collection_symbol}.columns', f'{table_symbol}_{column_symbol}.json')
    except Exception as e:
        return []

def get_collection_table_columns_spec_by_name(collection_name, name, symbols):
    """
    Get columns specification for a table in a collection.
    Args:
        collection_name (str): Name of the collection.
        collection_symbol (str): Symbol of the collection.
        table_name (str): Name of the table.
    Returns:
        list: List of column specifications.
    """
    table_spec = get_collection_table_spec_by_name(collection_name, symbols[-1], name)
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
            "symbols": column_spec.get('symbols', symbols + [table_spec['table_symbol'], column]),
            "ispassive": False,
            "default": column_spec.get('default', None)
        })
    
    return {'filters': answer}

def load_collection_filter_from_symbols(symbols):
    collection_config = get_collection_item_config(*symbols[:2])
    answer = {
        "title": collection_config['title'],
        "default": collection_config.get('default', ''),
        "items": get_collection_tables(collection_config),
        "children": collection_config.get('children', False),
        "selector_type": "combo_box",
        "symbols": symbols,
        "ispassive": False
    }

    return {"filters": [answer]}

def load_column_filter_from_symbols(symbols):
    collection_config = get_collection_item_config(*symbols[:2])
    table = symbols[2]
    table_spec = collection_config['tables'][table]
    column = symbols[3]
    column_spec = table_spec['columns'][column]
    mappings = table_spec.get('mappings', None)

    answer = {
            "title": column_spec['name'],
            "default": column_spec.get('default', ''),
            "items": get_items_for_column(column_spec, column, table, symbols[1], mappings),
            "children": column_spec.get('children', False),
            "selector_type": get_selector_type(column_spec['type']),
            "symbols": symbols,
            "ispassive": False,
            "default": column_spec.get('default', None)
        }
    
    return {"filters": [answer]}

def is_proper_name(name):
    if isinstance(name, str):
        if len(name) > 0:
            return True
    return False

def get_casual_filter_spec(symbols, name=None):

    collection_name = symbols[0] if len(symbols) > 0 else name
    
    if len(symbols) == 0:
        answer = get_collections_list(collection_name=collection_name)
        answer_item = answer['filters'][0]
        if len(answer_item['items']) == 1:
            symbols = answer_item['symbols']
            name = answer_item['items'][0]
        else:
            answer['filters'] = [get_subtitle_config(title='wybór obiektów')] + answer['filters']
            return answer
    
    if len(symbols) == 1:
        answer = get_collection_tables_by_name(collection_name=collection_name, name=name, symbols=symbols)
        answer_item = answer['filters'][0]
        if len(answer_item['items']) == 1:
            symbols = answer_item['symbols']
            name = answer_item['items'][0]
        else:
            return answer
        
    if len(symbols) == 2:
        answer = get_collection_table_columns_spec_by_name(collection_name=collection_name, name=name, symbols=symbols)
        if collection_name == 'collections':
            answer['filters'].extend([
                get_subtitle_config(title='kwalifikacja'),
                get_qualification_config()
            ])
        return answer
    
    print('Proper Name:', is_proper_name(name), name)

    if len(symbols) == 3:
        name = symbols[-1]
        answer = get_collection_table_columns_spec_by_name(collection_name=collection_name, name=name, symbols=symbols[:-1])
        print('ANSWER:', answer)
        return answer

    return None

def handle_symbols_special_cases(symbols):
    collection_name = symbols[0]
    collection_symbol = symbols[1]
    table_name = symbols[2]

    if (collection_name=='target') & (collection_symbol=='parcels') & (table_name=='łączenie działek ewidencyjnych'):
        print('\nSpecial case symbols', symbols[:2] + ['P', 'lacz_dz'])
        filter =  load_column_filter_from_symbols(symbols[:2] + ['P', 'lacz_dz'])
        filter['filters'][0]['symbols'] = symbols
        print('Special case filter', filter)
        return filter
    return None

def handle_filter_spec_reload(symbols):
    if len(symbols) == 0:
        return None
    elif (len(symbols) == 1) | (symbols[0] == 'subtitle'):
        collection_name = symbols[0]
        if collection_name not in ['qualification', 'subtitle']:
            return get_collections_list(collection_name=collection_name)
        else:
            if collection_name == 'qualification':
                return {'filters': [get_qualification_config()]}
            elif collection_name == 'subtitle':
                return {'filters': [get_subtitle_config(title=symbols[1])] }
            else:
                return None
    elif len(symbols) == 2:
        return load_collection_filter_from_symbols(symbols)
    elif len(symbols) == 3:
        return handle_symbols_special_cases(symbols)
    elif len(symbols) == 4:
        return load_column_filter_from_symbols(symbols)
    else:
        return None

    

def get_filter_spec(symbols, name=None):
    print(f'symbols: {symbols} | name: {name}')
    if symbols is None:
        return None
    elif name is not None:
        return get_casual_filter_spec(symbols=symbols, name=name)
    elif len(symbols) > 0:
        return handle_filter_spec_reload(symbols=symbols)
    else:
        return None
    
    
    
