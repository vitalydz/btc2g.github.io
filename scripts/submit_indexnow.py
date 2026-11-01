"""Notify participating search engines about public BTC2G page changes."""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = 'https://btc2g.com'
KEY_FILE = 'indexnow-key.txt'
MARKET_PAGES = {'', 'scenario.html', 'bitcoin-gold-ratio.html',
                'bitcoin-vs-gold.html', 'bitcoin-vs-gold-forecast-2026.html'}


def wait_for_publication(revision, opener=urlopen, sleep=time.sleep):
    url = f'https://api.github.com/repos/vitalydz/btc2g.github.io/actions/runs?head_sha={revision}&per_page=20'
    for _ in range(30):
        with opener(Request(url, headers={'User-Agent': 'BTC2G-IndexNow'}), timeout=20) as response:
            runs = json.load(response)['workflow_runs']
        pages = [run for run in runs if run['name'] == 'pages build and deployment']
        if pages and pages[0]['status'] == 'completed':
            if pages[0]['conclusion'] != 'success':
                raise RuntimeError('Pages publication did not succeed; no URLs submitted')
            return
        sleep(10)
    raise RuntimeError('Timed out waiting for Pages publication; no URLs submitted')


def sitemap_urls(root=ROOT):
    tree = ET.parse(root / 'sitemap.xml')
    urls = []
    for node in tree.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
        url = (node.text or '').strip()
        parts = urlsplit(url)
        if parts.scheme != 'https' or parts.netloc != 'btc2g.com' or parts.query or parts.fragment:
            raise ValueError('Sitemap contains a noncanonical or external URL')
        if url not in urls:
            urls.append(url)
    if not urls:
        raise ValueError('Sitemap is empty')
    return urls


def changed_urls(urls, files):
    files = set(files)
    if 'sitemap.xml' in files:
        return urls
    market_changed = any(path.startswith('assets/btc_') or path == 'assets/multi_asset_strength.json' for path in files)
    shared_changed = any(path.startswith(('assets/js/', 'assets/css/')) for path in files)
    selected = []
    for url in urls:
        page = urlsplit(url).path.lstrip('/')
        local = page or 'index.html'
        if local in files or shared_changed or (market_changed and page in MARKET_PAGES):
            selected.append(url)
    return selected


def submit(urls, root=ROOT, opener=urlopen):
    if not urls:
        print('No changed public URLs to submit.')
        return None
    if len(urls) > 10000 or any(url not in sitemap_urls(root) for url in urls):
        raise ValueError('Only canonical public sitemap URLs may be submitted')
    key = (root / KEY_FILE).read_text(encoding='utf-8').strip()
    if not re.fullmatch(r'[a-zA-Z0-9-]{8,128}', key):
        raise ValueError('Invalid IndexNow verification key')
    key_url = f'{ORIGIN}/{KEY_FILE}'
    # Never notify engines before the verification file is actually deployed.
    with opener(Request(key_url, headers={'Cache-Control': 'no-cache'}), timeout=20) as response:
        if response.read().decode('utf-8').strip() != key:
            raise RuntimeError('Published IndexNow verification file does not match')
    payload = {'host': 'btc2g.com', 'key': key, 'keyLocation': key_url, 'urlList': urls}
    request = Request('https://api.indexnow.org/indexnow',
                      data=json.dumps(payload).encode('utf-8'), method='POST',
                      headers={'Content-Type': 'application/json; charset=utf-8'})
    with opener(request, timeout=30) as response:
        status = response.status
    if status not in (200, 202):
        raise RuntimeError(f'IndexNow returned HTTP {status}')
    print(f'IndexNow received {len(urls)} URLs (HTTP {status}); indexing is not guaranteed.')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Initial discovery of all sitemap URLs')
    parser.add_argument('--revision', default='HEAD', help='Deployed commit to inspect')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--wait-for-publication', action='store_true')
    args = parser.parse_args()
    revision = subprocess.check_output(['git', 'rev-parse', '--verify', args.revision + '^{commit}'], cwd=ROOT, text=True).strip()
    urls = sitemap_urls()
    if not args.all:
        files = subprocess.check_output(['git', 'diff-tree', '--root', '--no-commit-id', '--name-only', '-r', revision], cwd=ROOT, text=True).splitlines()
        urls = changed_urls(urls, files)
    if args.dry_run:
        print(json.dumps(urls, indent=2))
    else:
        if args.wait_for_publication:
            wait_for_publication(revision)
        submit(urls)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'IndexNow submission failed: {exc}', file=sys.stderr)
        sys.exit(1)
