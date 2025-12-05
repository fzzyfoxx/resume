from geodoc_run.src.queue import add_teryts_to_queue
import argparse

def main():
    """
    Runs the add_teryts_to_queue function to add TERYT codes to the processing queue based on provided arguments:
     - service: Name of the service to which the task is added.
     - source_type: Type of the source data (default: administration_units).
     - source_key: Key for the specific source table.
     - teryt_pattern: TERYT code pattern to match.
     - priority: Priority of the task in the queue (default: 1).
    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Add a TERYT code to the processing queue.")
    parser.add_argument('--service', type=str, required=True, help='Name of the service to which the task is added.')
    parser.add_argument('--source_type', type=str, default='administration_units', required=False, help='Type of the source data (e.g., administration_units).')
    parser.add_argument('--source_key', type=str, required=True, help='Key for the specific source table.')
    parser.add_argument('--teryt_pattern', type=str, required=True, help='TERYT code pattern to match.')
    parser.add_argument('--priority', type=int, default=1, required=False, help='Priority of the task in the queue.')

    args = parser.parse_args()

    add_teryts_to_queue(args.service, args.source_type, args.source_key, args.teryt_pattern, args.priority)

if __name__ == "__main__":
    main()