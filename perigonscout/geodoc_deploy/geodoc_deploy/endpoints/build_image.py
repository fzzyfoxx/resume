import argparse
from geodoc_deploy.deploy.build_image_src import build_image

def main():
    parser = argparse.ArgumentParser(description="Deploy a GeoDoc service.")
    parser.add_argument("--service", type=str, required=True, help="The name of the service to deploy.")
    
    args = parser.parse_args()
    
    # Deploy the specified service
    build_image(args.service)

if __name__ == "__main__":
    main()