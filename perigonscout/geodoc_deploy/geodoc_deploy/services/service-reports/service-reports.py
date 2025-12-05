from geodoc_config import get_service_config
import os

if __name__ == "__main__":
    # Load the example service configuration
    service = str(os.environ.get("SERVICE", None))
    key = str(os.environ.get("SERVICE_KEY", None))
    config = get_service_config(service, key)
    
    # Print the configuration for debugging purposes
    print(f"Loaded configuration for service '{service}/{key}':")
    print(config)
    
    # Here you can add code to start the dummy service using the loaded configuration
    # For example, initializing a web server or a background task