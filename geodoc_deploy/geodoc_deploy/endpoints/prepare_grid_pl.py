import argparse
from geodoc_loader.services.grid import prepare_and_load_grid
from geodoc_config import get_service_config

def main():
    parser = argparse.ArgumentParser(description="Prepare and load grid data to BigQuery.")
    parser.add_argument('--grid_size', type=int, required=True, help='Size of the grid in meters.')
    parser.add_argument('--min_side_buffer', type=int, required=True, help='Minimum side buffer in meters.')
    parser.add_argument('--delete_local', type=int, default=1, help='Delete local temporary files after processing.')

    args = parser.parse_args()

    config = get_service_config('grid', 'worker')

    success = prepare_and_load_grid(
        grid_size=args.grid_size,
        min_side_buffer=args.min_side_buffer,
        config=config,
        delete_local=bool(args.delete_local)
    )

    if success:
        print("Grid preparation and loading completed successfully.")
    else:
        print("An error occurred during grid preparation or loading.")

if __name__ == "__main__":
    main()