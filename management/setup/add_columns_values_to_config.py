from geodoc_app.inputs.values import get_collection_item_config, get_collections_list, COLLECTIONS_CONFIG
from geodoc_app.search.search_queries import prepare_table_filter_query, distinct_table_column_values_query
from geodoc_config import load_config_by_path
from google.cloud import bigquery
from tqdm.auto import tqdm
import os
import json
import argparse

def main():

    parser = argparse.ArgumentParser(description='Generate filter values for a specific table and column.')
    parser.add_argument('--main_path', type=str, default='./geodoc_config/geodoc_config/configs/search', help='Path to the main configuration directory.')
    parser.add_argument('--folder_name', type=str, default='columns', help='Name of the folder to save the filter values.')

    args = parser.parse_args()
    main_path = args.main_path
    folder_name = args.folder_name

    bq_client = bigquery.Client()
    project_id = bq_client.project

    collections_num = len(COLLECTIONS_CONFIG['collections'])

    for i, (collection_symbol, collection_def) in enumerate(COLLECTIONS_CONFIG['collections'].items()):
        print(f"Processing collection: {collection_symbol} ({i + 1}/{collections_num})")
        collection_config = get_collection_item_config(collection_symbol)
        dataset_id = collection_config['dataset_id']

        # count items for progress bar
        total_items = sum([1 for table in collection_config['tables'].values() for column_spec in table['columns'].values() if column_spec['type'] in ['CATEGORICAL', 'SEARCH']])
        progress_bar = tqdm(total=total_items, desc=f"Processing collection {collection_symbol}")

        for table_symbol, table_config in collection_config['tables'].items():

            for column, column_spec in table_config['columns'].items():

                if column_spec['type'] not in ['CATEGORICAL', 'SEARCH']:
                    #print(f"Skipping column {column} in table {table_symbol} - not a CATEGORICAL or SEARCH type ({column_spec['type']}).")
                    continue

                table_query = distinct_table_column_values_query(
                    table_spec=table_config,
                    column=column,
                    project_id=project_id,
                    dataset_id=dataset_id
                    )
                
                query_job = bq_client.query(table_query)
                results = query_job.result()
                parsed_results = [dict(row)['value'] for row in results]

                save_folder = os.path.join(main_path, collection_symbol, folder_name)
                file_path = os.path.join(save_folder, f'{table_symbol}_{column}.json')
                file_content = {'column': column, 'values': parsed_results}
                os.makedirs(save_folder, exist_ok=True)

                with open(file_path, 'w', encoding='utf=8') as f:
                    json.dump(file_content, f, indent=4, ensure_ascii=False)

                #print(f"Saved filter values for {collection_symbol} - {table_symbol} - {column} to {file_path}")
                progress_bar.update(1)  # Update progress bar after processing each column

            #print(f"Processed table {table_symbol} in collection {collection_symbol}")

        print(f"Processed collection {collection_symbol}")

    print("All filter values generated successfully.")

if __name__ == "__main__":
    main()