from google.cloud import storage
from typing import List, Dict


def list_png_files_and_upload(bucket_name, prefixes):
    """List all .png files in the GCS bucket directory, save them to a .txt file, and upload the file to another GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    json_file_per_subset = list_json_files(bucket_name, prefixes)
    for prefix in prefixes:
        blobs = bucket.list_blobs(prefix=prefix)  # List all objects that start with the prefix.

        png_files = [blob.name for blob in blobs if blob.name.endswith('.png')]
        json_files = json_file_per_subset[prefix]
        # Extract the base names without the extension
        json_basenames = [file.split('.')[0] for file in json_files]

        # Filter out image names that do not have a corresponding JSON file
        filtered_png_files = [file for file in png_files if file.split('.')[0] in json_basenames]

        # Write the .png files to a temporary local file
        temp_file_path = '/tmp/png_files_list.txt'
        with open(temp_file_path, 'w') as file:
            for line in filtered_png_files:
                line = line.split('/')[-1]  # Get the file name only
                file.write(line + '\n')

        # Upload the .txt file to the destination bucket
        destination_blob_name = f'{prefix.replace("/", "")}_images_names.txt'
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(temp_file_path)

        print(f'List of .png files has been uploaded to gs://{bucket}/{destination_blob_name}')


def list_json_files(bucket_name, prefixes) -> Dict[str, List[str]]:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    json_files = dict()
    for prefix in prefixes:
        blobs = bucket.list_blobs(prefix=prefix)  # List all objects that start with the prefix.

        prefix_json_files = [blob.name for blob in blobs if blob.name.endswith('.json')]
        json_files[prefix] = prefix_json_files
    return json_files


source_bucket_name = 'effizency-datasets'
prefixes = ['train/', 'val/']  # Make sure to include the trailing slash if you want to list a directory

list_png_files_and_upload(source_bucket_name, prefixes)
