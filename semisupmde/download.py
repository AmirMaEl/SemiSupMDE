"""Download the models and the data needed for evaluation or finetuning.

  semisupmde download models           # pretrained and finetuned checkpoints into CHECKPOINT_DIR
  semisupmde download kitti test       # KITTI files of the given splits (test, val, train) into KITTI_ROOT
  semisupmde download vkitti           # Virtual KITTI 1.3.1 RGB and depth into VKITTI_ROOT (about 20 GB)

The KITTI raw drives are zip archives of up to several GB on the KITTI server. Only the files listed in the
split files are needed (the image, plus the velodyne scan for the evaluation splits), so these are read
directly out of the remote archives with HTTP range requests instead of downloading whole drives
(about 2 GB for the test split instead of 76 GB). Everything is skipped if it already exists.
"""
import hashlib
import io
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config

KITTI_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'
VKITTI_URL = 'https://download.europe.naverlabs.com/virtual-kitti-1.3.1/'
RETRIES = 5


def urlopen(url, headers=None, method=None):
    """urlopen with retries and exponential backoff."""
    for attempt in range(RETRIES):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}, method=method),
                                          timeout=60)
        except OSError as e:
            # client errors such as 404 will not go away by retrying (except 429, too many requests)
            permanent = isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429
            if permanent or attempt == RETRIES - 1:
                raise
            print('  %s: %s, retrying' % (url, e))
            time.sleep(2 ** (attempt + 1))


def download(url, path):
    """Download url to path, resuming a partial download from path.part."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.part'
    size = int(urlopen(url, method='HEAD').headers['Content-Length'])
    while not os.path.exists(tmp) or os.path.getsize(tmp) < size:
        done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        print('downloading %s (%.1f / %.1f GB)' % (os.path.basename(path), done / 1e9, size / 1e9))
        try:
            with urlopen(url, {'Range': 'bytes=%d-' % done}) as r, open(tmp, 'ab') as f:
                shutil.copyfileobj(r, f, 1 << 22)
        except OSError as e:
            print('  interrupted (%s), resuming' % e)
            time.sleep(2)
    os.rename(tmp, path)


class RemoteFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests (enough for zipfile)."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(urlopen(url, method='HEAD').headers['Content-Length'])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        for attempt in range(RETRIES):
            try:
                with urlopen(self.url, {'Range': 'bytes=%d-%d' % (self.pos, end)}) as r:
                    chunk = r.read()
                if len(chunk) == end - self.pos + 1:
                    break
            except OSError:
                pass
            time.sleep(2 ** (attempt + 1))
        else:
            raise IOError('failed to read %s' % self.url)
        buf[:len(chunk)] = chunk
        self.pos += len(chunk)
        return len(chunk)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def download_model(name):
    """Download a released model ('pretrained' or 'finetuned') if needed and return its path."""
    filename, urls, checksum = config.MODELS[name]
    path = os.path.join(config.CHECKPOINT_DIR, filename)
    if os.path.exists(path):
        return path
    for url in urls:
        try:
            download(url, path)
        except OSError as e:
            print('  %s failed (%s)' % (url, e))
            continue
        if md5(path) == checksum:
            return path
        print('  %s does not have the expected md5 %s' % (url, checksum))
        os.remove(path)
    raise IOError('could not download %s from %s' % (filename, ', '.join(urls)))


def download_models():
    for name in config.MODELS:
        download_model(name)


def kitti_files(splits):
    """{drive: [paths relative to the KITTI root]} of all files the given splits need."""
    files = {}
    for split in splits:
        with open(config.SPLITS[split]) as f:
            images = [line.strip() for line in f if line.strip()]
        for img in images:
            if split == 'train':
                # unlabelled target images, read from the finetuning camera
                needed = [img.replace('image_02', config.FINETUNE['target_camera'])]
            else:
                needed = [img, img.replace('image_02', 'velodyne_points').replace('.png', '.bin')]
            drive = img.split('/')[1][:-len('_sync')]
            files.setdefault(drive, set()).update(needed)
    return {drive: sorted(paths) for drive, paths in files.items()}


def fetch_files(drive, paths, root):
    """Extract the given files of a KITTI drive out of the remote zip archive."""
    remote = RemoteFile('%s%s/%s_sync.zip' % (KITTI_URL, drive, drive))
    with zipfile.ZipFile(io.BufferedReader(remote, 1 << 21)) as z:
        for p in paths:
            out = os.path.join(root, p)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out + '.part', 'wb') as f:
                f.write(z.read(p))
            os.rename(out + '.part', out)
    return len(paths)


def download_kitti(splits, workers=16, chunk_size=500):
    root = config.KITTI_ROOT
    files = kitti_files(splits)
    # camera calibration of every recording date
    for date in sorted({drive[:10] for drive in files}):
        if not os.path.exists(os.path.join(root, date, 'calib_cam_to_cam.txt')):
            zip_path = os.path.join(root, date + '_calib.zip')
            download(KITTI_URL + date + '_calib.zip', zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(root)
            os.remove(zip_path)

    # files that are not there yet, in chunks so that large drives are read in parallel
    chunks = []
    for drive, paths in files.items():
        missing = [p for p in paths if not os.path.exists(os.path.join(root, p))]
        chunks += [(drive, missing[i:i + chunk_size]) for i in range(0, len(missing), chunk_size)]
    n_missing = sum(len(c) for _, c in chunks)
    print('KITTI %s: %d files from %d drives in %s, %d to download'
          % ('+'.join(splits), sum(len(p) for p in files.values()), len(files), root, n_missing))
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for job in as_completed([pool.submit(fetch_files, d, c, root) for d, c in chunks]):
            done += job.result()
            print('  %d / %d files' % (done, n_missing), flush=True)


def download_vkitti():
    root = config.VKITTI_ROOT
    for name in ('vkitti_1.3.1_rgb', 'vkitti_1.3.1_depthgt'):
        if os.path.isdir(os.path.join(root, name)):
            continue
        tar_path = os.path.join(root, name + '.tar')
        download(VKITTI_URL + name + '.tar', tar_path)
        print('extracting %s' % tar_path)
        tmp = os.path.join(root, '.extract_' + name)
        with tarfile.open(tar_path) as t:
            if hasattr(tarfile, 'data_filter'):
                t.extractall(tmp, filter='data')
            else:
                t.extractall(tmp)
        os.rename(os.path.join(tmp, name), os.path.join(root, name))
        os.rmdir(tmp)
        os.remove(tar_path)
