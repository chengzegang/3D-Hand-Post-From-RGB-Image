from tqdm import tqdm
import urllib.request
import zipfile
import math

data_url = 'https://lmb.informatik.uni-freiburg.de/data/RenderedHandpose/RHD_v1-1.zip'
filename = 'RHD.zip'
data_folder = './data/'

pbar = None
last_pbar = 0


def show_progress(block_num, block_size, total_size):
    global pbar, last_pbar

    if pbar is None:
        pbar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                    desc="Downloading Rendered Handpose Dataset")

    downloaded = block_num * block_size
    if downloaded < total_size:
        if downloaded != last_pbar:
            pbar.update(downloaded - last_pbar)
            last_pbar = downloaded
    else:
        pbar.close()
        pbar = None
        last_pbar = 0


urllib.request.urlretrieve(data_url, filename, show_progress)

with zipfile.ZipFile(filename, 'r') as zip_ref:
    for file in tqdm(zip_ref.infolist(), desc='Extracting'):
        try:
            zip_ref.extract(member=file, path=data_folder)
        except zip_ref.error as e:
            pass
