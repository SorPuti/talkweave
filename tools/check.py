"""Check Pages paths, downloads, and agent manifests without external dependencies."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'docs'
BASE='https://sorputi.github.io/talkweave/'
checked=0

class Links(HTMLParser):
    """Collect navigation and asset links from static HTML."""
    def __init__(self):
        """Initialize a local list of discovered links."""
        super().__init__();self.links=[]
    def handle_starttag(self, tag, attrs):
        """Record href and src attributes with nonempty values."""
        self.links.extend(value for key,value in attrs if key in {'href','src'} and value)

for page in SITE.rglob('*.html'):
    parser=Links();parser.feed(page.read_text())
    page_url=BASE+page.relative_to(SITE).as_posix()
    for link in parser.links:
        if link.startswith(('data:','javascript:','mailto:','#')):continue
        url=urlsplit(urljoin(page_url,link));base=urlsplit(BASE)
        if url.netloc!=base.netloc:continue
        assert url.path.startswith(base.path),f'Link escaped Pages subpath: {page}: {link}'
        target=SITE/unquote(url.path[len(base.path):])
        if url.path.endswith('/'):target=target/'index.html'
        assert target.is_file(),f'Broken local link: {page}: {link} -> {target}'
        checked+=1
plan=json.loads((SITE/'agents/plan.json').read_text())
package=SITE/'downloads/talkweave.zip'
assert hashlib.sha256(package.read_bytes()).hexdigest()==plan['package']['sha256']
for record in plan['files']:
    path=ROOT/'lab'/record['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],record['path']
    parsed=urlsplit(record['url'])
    assert parsed.path.startswith('/talkweave/')
    assert (SITE/parsed.path.removeprefix('/talkweave/')).is_file(),record['url']
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None
    for name in ['run.py','src/core.py','src/infer.py','AGENTS.md','.agents/skills/port/SKILL.md','rs/src/lib.rs','docs/port.md']:
        assert 'talkweave/'+name in z.namelist(),name
assert (SITE/'.nojekyll').exists()
for page in [SITE/'agents/index.html',SITE/'agents/guide.md',SITE/'llms.txt']:
    assert 'chatgpt.site' not in page.read_text()
print(json.dumps({'pages_links_checked':checked,'zip':'valid','package_sha256':'matches','agent_files':len(plan['files']),'pages_subpath':'valid'}))
