import docker
import google.auth
from google.auth import impersonated_credentials
from google.auth.transport.requests import Request
import subprocess

def authenticate_and_push_image(
    image_tag: str,
    service_name: str,
    service_account_email: str,
    artifact_registry_host: str = "europe-west1-docker.pkg.dev"
):
    """
    Authenticates to Google Artifact Registry using service account impersonation
    and pushes a Docker image.

    Args:
        image_tag (str): The full tag of the Docker image to push (e.g., "europe-west1-docker.pkg.dev/my-project/my-repo/my-image:latest").
        project_id (str): Your Google Cloud project ID.
        service_account_email (str): The email of the service account to impersonate
                                     (e.g., "geodoc-repo@your-project.iam.gserviceaccount.com").
        artifact_registry_host (str): The Artifact Registry hostname.
                                      Defaults to europe-west1-docker.pkg.dev.
    """

    access_token = subprocess.run(
        [
            "gcloud", "auth", "print-access-token",
            "--impersonate-service-account", service_account_email
        ],
        stdout=subprocess.PIPE,
        check=True
    ).stdout.decode("utf-8").strip()

    subprocess.run(
        [
            "docker", "login",
            "-u", "oauth2accesstoken",
            "--password-stdin", artifact_registry_host
        ],
        input=access_token.encode(),  # Pass the access token to stdin
        check=True
    )
    
    print("Connecting to Docker client...")
    client = docker.from_env()
    print("Tag service image for Artifact Registry...")
    # get docker image for service
    service_image = client.images.get(f"{service_name}:latest")

    # add a tag to the image for Artifact Registry
    service_image.tag(image_tag)

    print(f"Authenticating to Artifact Registry: {artifact_registry_host}...")

    # 1. Get current credentials (e.g., from gcloud auth application-default login,
    #    GOOGLE_APPLICATION_CREDENTIALS env var, or attached service account if on GCP)
    #    These are the credentials of the entity *performing* the impersonation.
    try:
        source_credentials, _ = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        print("Error: Could not find default Google Cloud credentials. "
              "Please run 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS.")
        return

    # 2. Impersonate the service account to get short-lived credentials (access token)
    #    The `source_credentials` must have the `roles/iam.serviceAccountTokenCreator`
    #    role on the `service_account_email` to be impersonated.
    try:
        impersonated_creds = impersonated_credentials.Credentials(
            source_credentials=source_credentials,
            target_principal=service_account_email,
            target_scopes=["https://www.googleapis.com/auth/cloud-platform"], # Broad scope needed for pushing Docker images
            lifetime=3600  # Token valid for 1 hour
        )
        impersonated_creds.refresh(Request()) # Request the token
        access_token = impersonated_creds.token
        print(f"Successfully obtained impersonated access token.")
    except Exception as e:
        print(f"Error during service account impersonation: {e}")
        print("Ensure your current authentication has 'Service Account Token Creator' role on the target service account.")
        return

    # 3. Perform Docker login using the access token
    try:
        login_result = client.login(
            username='oauth2accesstoken',
            password=access_token,
            registry=artifact_registry_host
        )
        if login_result and 'Status' in login_result:
            print(f"Docker login status: {login_result['Status']}")
        else:
            print("Docker login successful (status not explicitly returned).")
    except docker.errors.APIError as e:
        print(f"Docker login failed: {e}")
        return

    # 4. Push the image
    print(f"Pushing image {image_tag} to Artifact Registry...")
    try:
        push_logs = client.images.push(repository=image_tag, stream=True, decode=True)
        for line in push_logs:
            if 'status' in line:
                if line['status']== 'Pushed':
                    print(f"Successfully pushed image: {image_tag}")
            elif 'error' in line:
                print(f"Push Error: {line['error']}")
    except docker.errors.APIError as e:
        print(f"Docker push failed: {e}")

