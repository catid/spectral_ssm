import os, time, requests
import torch
import concurrent.futures
import tarfile

# Downloads a file: Implements retries
def download_file(url, destination_folder, tmp_file_name, max_retries=3):
    retries = 0
    os.makedirs(destination_folder, exist_ok=True)
    tmp_file_path = os.path.join(destination_folder, tmp_file_name)

    while retries < max_retries:
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()

                with open(tmp_file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"Downloaded {url} to {tmp_file_path}")
            return
        except requests.RequestException as e:
            retries += 1
            print(f"Failed to download {url}, attempt {retries}/{max_retries}. Error: {e}")
            time.sleep(2 ** retries)  # Exponential backoff

    raise Exception(f"Failed to download {url} after {max_retries} attempts.")

def download_and_extract_multipart_tar(urls, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    parts = []  # Initialize an empty list to store the paths of downloaded parts

    # Use ThreadPoolExecutor to download in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as executor:
        # Submit download tasks
        future_to_part = {executor.submit(download_file, url, destination_folder, f"temp_{idx}.tar"): idx for idx, url in enumerate(urls)}

        # Wait for each download to complete and collect part paths
        for future in concurrent.futures.as_completed(future_to_part):
            idx = future_to_part[future]
            part_filename = f"temp_{idx}.tar"
            part_path = os.path.join(destination_folder, part_filename)
            try:
                # Ensure the download was successful before adding the part path
                future.result()
                parts.append(part_path)  # Add the successfully downloaded part path
            except Exception as exc:
                print(f"Download failed for part {idx}: {exc}")

    combined_tar = os.path.join(destination_folder, "combined.tar")
    print(f"Downloads completed. Combining parts into {os.path.basename(combined_tar)}")

    parts.sort()

    # Combine the parts into a single tar file
    with open(combined_tar, 'wb') as wfd:
        for part in parts:
            with open(part, 'rb') as fd:
                wfd.write(fd.read())

    # Optionally, extract the combined tar file
    print(f"Extracting {os.path.basename(combined_tar)}")
    with tarfile.open(combined_tar) as tar:
        tar.extractall(path=destination_folder)

    parts.append(combined_tar)
    print(f"parts = {parts}")
    for part in parts:
        os.remove(part)

    print("Extraction completed")

urls = [
    'https://www.dropbox.com/scl/fi/y8nbpkmtwtqilzu37gh2n/doom-original-game-soundtrack.tar.001?rlkey=1isq3dgwtxxg6n9lr0ad7y5zl&dl=1',
    'https://www.dropbox.com/scl/fi/ymggk133agw5k2knn0j6q/doom-original-game-soundtrack.tar.002?rlkey=e5on690fmsb6b7zbyt44rxipz&dl=1',
    'https://www.dropbox.com/scl/fi/rqlqoch1gp81qvilhajf2/doom-original-game-soundtrack.tar.003?rlkey=4bfyd6iv79xjw8pkmxhhyse45&dl=1',
    'https://www.dropbox.com/scl/fi/prkzzuyzc0k9rc4kab00t/doom-original-game-soundtrack.tar.004?rlkey=gffakparmoybxsf2te939zd9i&dl=1',
    'https://www.dropbox.com/scl/fi/d5hge7cbcv0d4ptbk54ly/doom-original-game-soundtrack.tar.005?rlkey=jbsgxxiexzpfxsob0ogyl0oxl&dl=1',
]

destination_folder = './data'
download_and_extract_multipart_tar(urls, destination_folder)
