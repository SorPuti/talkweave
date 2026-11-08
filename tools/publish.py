"""Push the prepared repository and configure classic GitHub Pages when authorized."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REPO='SorPuti/talkweave'


def run(args, payload=None, required=True):
    """Execute a command without exposing authentication values."""
    result=subprocess.run(args,cwd=ROOT,input=json.dumps(payload) if payload is not None else None,text=True,capture_output=True)
    if result.returncode and required:
        print(result.stderr.strip(),file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def main():
    """Publish only to the selected repository and report the actual Pages state."""
    user=run(['gh','api','user','--jq','.login']).stdout.strip()
    if user.lower()!='sorputi':
        raise SystemExit('Authenticate the intended SorPuti account before publishing.')
    repo=run(['gh','api','repos/'+REPO],required=False)
    if repo.returncode:
        raise SystemExit('Create the empty public repository first: https://github.com/new?name=talkweave')
    run([sys.executable,'build.py'])
    run([sys.executable,'tools/check.py'])
    run(['git','add','.'])
    changed=run(['git','diff','--cached','--quiet'],required=False)
    if changed.returncode:
        run(['git','commit','-m','Prepare TalkWeave tutorial, executable lab and GitHub Pages downloads'])
    run(['git','push','-u','origin','main'])
    pages=run(['gh','api','repos/'+REPO+'/pages'],required=False)
    if pages.returncode:
        pages=run(['gh','api','--method','POST','repos/'+REPO+'/pages','--input','-'],{'source':{'branch':'main','path':'/docs'}},required=False)
    if pages.returncode:
        print('Source pushed. Pages configuration requires repository permission or a manual setting:')
        print('https://github.com/'+REPO+'/settings/pages')
        print('Deploy from a branch -> main -> /docs -> Save')
        return 2
    data=json.loads(pages.stdout)
    print(json.dumps({'repository':'https://github.com/'+REPO,'pages':data.get('html_url'),'status':data.get('status'),'source':data.get('source')},indent=2))
    print('Confirm the Pages deployment is built before claiming publication succeeded.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
