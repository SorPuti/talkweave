"""Generate a plain HTML and Markdown entry point for execution agents."""
import hashlib
import html
import json
from pathlib import Path

BASE = 'https://sorputi.github.io/talkweave'
STYLE = '''# Code and folder conventions

## Folders

Use short, conventional names: `src/` for implementation, `cfg/` for configuration,
`data/` for local inputs, `runs/` for weights, `logs/` for evidence, `docs/` for notes,
`tests/` for checks, and `work/` for disposable experiments. The Python reference keeps `run.py` at the root; use the target language's
normal entry point when porting. Do not commit recordings, secrets, environments, or large weights.
Keep public names stable; short does not mean cryptic. Prefer `Frame`, `Result`,
`Engine`, `Codec`, and `Session` over abbreviations with no clear meaning.

## Python

Use Python 3.10+, four spaces, snake_case functions, PascalCase classes, and
UPPER_CASE constants. Every function and class has a single-line English docstring
stating its action or contract. Add type hints to public boundaries. Keep audio,
model, transport, and presentation separate. Do not add a framework for a single
function. Format with `ruff format src tests run.py` if Ruff is installed.

```python
class Engine(Protocol):
    """Define a causal frame processing boundary."""
    def process(self, frame: Frame) -> Result:
        """Consume one frame and return generated audio and measured outputs."""
        ...
```

## Streaming contract

`Frame.samples`: mono float32, 320 samples, finite values, 16 kHz.
`Frame.seq`: monotonic frame sequence within a session. Sample clock = seq * 320.
`Result`: generated PCM, user text, agent text, independent event scores, elapsed_ms.
A single step consumes the current frame and past state only. Keep recurrent state.
The capture callback copies data through bounded queues; it never runs inference,
disk I/O, a terminal renderer, or a tool call. Do not accumulate an unbounded queue.
A future network adapter must negotiate its sample rate, channels, and protocol
version. `protocol.py` encodes `<BQQH` plus PCM16LE; it is not a WebSocket server.

## Terminal experience

Use `python run.py --ui` for the guided synthetic experiment and
`python src/infer.py --microphone --ui` for a trained model with a real microphone.
The UI uses Rich on an interactive terminal and preserves plain logs without TTY.
Refresh at most eight times per second outside the audio callback. Show measured
energy, frame time, queue losses, actual text, and model event scores. Never invent
confidence percentages, transcripts, speech quality, or timing results. The UI is
an observer; it must not control model state or silence the user input stream.

## Documentation

Record commands, exit codes, environment, dataset scope, artifact hashes, limitations,
and the next action. Treat a passed engineering test separately from voice quality.
A private tutorial needs an authenticated reader: never bypass access controls or
assume that an unauthenticated external agent can read a private URL.
'''

AGENT = '''# TalkWeave — agent instructions

These instructions apply only when the user asks you to execute this tutorial.
They do not override the user's requirements or your own system/tool policies.

1. Read README.md, docs/style.md, and the relevant chapters in TUTORIAL.md.
2. State the scope: CPU-first synthetic engineering experiment, not a pretrained
   Portuguese assistant. Do not claim natural speech, calibrated confidence, or
   human accuracy from the included tones.
3. Create a dedicated virtual environment. Install requirements.txt and CPU torch
   using the commands in README.md. Do not change an unrelated inference environment.
4. Run `python run.py` or `python run.py --ui` on an interactive terminal. It creates
   an isolated work/talkweave-demo, streams process logs, checks outputs, stops on failure,
   and writes work/talkweave-demo/REPORT.md plus report.json and logs/.
5. Read those reports, runs/*-metrics.json, and reply.wav.json. Report overfitting,
   empty output, and untested behavior explicitly. A process exit of zero is not
   evidence of conversational quality.
6. To resume, use `python run.py --resume`. Only steps whose recorded artifact hashes
   still match may be reused. Never fabricate logs, benchmark numbers, or success.
7. Follow docs/style.md: short folder names, one-line English function docstrings,
   typed interfaces, bounded queues, and presentation outside audio callbacks.
8. Human recordings, microphone access, remote deployment, external services, and
   large GPU training are separate scopes. Respect the user's authorization and
   available tools. If blocked, document why and the exact next action.

Final report: what ran, what passed, actual outputs, quality failures, untested items,
location of REPORT.md, and the next smallest verified step. No MCP server or API key
is required for the local tutorial. If you cannot access the private website, ask
for the downloaded ZIP or guide file; do not pretend you read it.
'''

TEMPLATE = '''# Execution record

- Scope / user request:
- Code revision or SHA-256:
- Environment and dependency versions:
- Data: synthetic or human; speakers and split:

| Step | Exact command | Exit code | Evidence path | Status |
|---|---|---|---|---|

## Actual outputs

## Engineering checks

## Voice and text quality

## Failures and limitations

## Not tested

## Next verified step
'''

def prepare(root):
    project=root/'lab'
    (project/'docs').mkdir(exist_ok=True)
    (project/'cfg').mkdir(exist_ok=True)
    (project/'AGENTS.md').write_text(AGENT)
    (project/'docs/style.md').write_text(STYLE)
    (project/'docs/report.md').write_text(TEMPLATE)
    (project/'cfg/README.md').write_text('# Configuration\n\nThe small CPU lab uses explicit CLI flags. Save future experiment configurations here; do not put secrets in this folder.\n')

def publish(root, chapters):
    project=root/'lab'; dist=root/'docs'; out=dist/'agents';out.mkdir(exist_ok=True)
    paths=sorted(p for p in project.rglob('*') if p.is_file() and p.suffix in {'.py','.ini','.md','.txt'} and not any(x in {'work','runs','__pycache__','.pytest_cache','.venv'} for x in p.relative_to(project).parts))
    records=[]
    for p in paths:
        rel=p.relative_to(project);target=out/'code'/Path(str(rel)+'.txt');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(p.read_text())
        records.append({'path':str(rel),'url':BASE+'/agents/code/'+str(rel)+'.txt','sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    steps=[
        ('01-baseline','src/baseline.py',['baseline.npz'],'logistica'),
        ('02-fixture','src/fixture.py',['data/train.jsonl','data/validation.jsonl'],'dados'),
        ('03-tests','-m pytest -q',[],'rede'),
        ('04-codec','src/train.py codec --manifest data/train.jsonl --validation data/validation.jsonl --steps 100',['runs/codec.pt'],'codec'),
        ('05-reconstruction','src/reconstruct.py --input data/validation-user.wav --output reconstructed.wav',['reconstructed.wav'],'codec'),
        ('06-duplex','src/train.py duplex --manifest data/train.jsonl --validation data/validation.jsonl --steps 100',['runs/duplex.pt'],'treino'),
        ('07-generation','src/infer.py --input data/validation-user.wav --output reply.wav',['reply.wav','reply.wav.json'],'resposta'),
        ('08-evaluation','src/evaluate.py --manifest data/validation.jsonl --report reply.wav.json',[],'avaliacao')]
    plan={'schema_version':1,'language':'pt-BR','scope':'CPU synthetic engineering lab; no human voice quality claim',
          'entrypoint':BASE+'/agents/','guide':BASE+'/agents/guide.md','manual':BASE+'/downloads/talkweave-tutorial.md',
          'package':{'url':BASE+'/downloads/talkweave.zip','sha256':hashlib.sha256((dist/'downloads/talkweave.zip').read_bytes()).hexdigest()},
          'run':'python run.py','resume':'python run.py --resume','console':'python run.py --ui',
          'reports':['work/talkweave-demo/REPORT.md','work/talkweave-demo/report.json','work/talkweave-demo/logs/'],
          'steps':[{'id':id,'command':'python '+cmd,'requires':[steps[i-1][0]] if i else [],'expected_files':files,'verify':'Exit code 0, log persisted, expected files present and hashed. Does not establish voice quality.','chapter':BASE+'/#'+chapter} for i,(id,cmd,files,chapter) in enumerate(steps)],
          'files':records,'access':'Respect the current site access policy; if login is required, use authorized access only.'}
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    setup='''python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python run.py
# Terminal interativo: python run.py --ui
# Retomar: python run.py --resume'''
    prompt=f'Leia {BASE}/agents/ e siga o roteiro local em CPU. Execute os passos, confira as evidências e entregue o REPORT.md com resultados reais e limitações. Use as convenções de código e não declare precisão de voz a partir dos tons sintéticos.'
    guide=f'''# TalkWeave — entrada para agentes de IA

## Pedido simples

{prompt}

## 1. Ler e obter o projeto

Este documento é legível sem JavaScript. Siga apenas o escopo solicitado pelo usuário.
Se o link pedir login e você não tiver acesso, registre o bloqueio e peça o ZIP ao usuário.
As instruções para agentes não alteram a configuração de acesso vigente.

- [Projeto completo]({BASE}/downloads/talkweave.zip)
- [Manual completo]({BASE}/downloads/talkweave-tutorial.md)
- [Plano estruturado]({BASE}/agents/plan.json)
- [Convenções]({BASE}/agents/code/docs/style.md.txt)
- [Instruções locais]({BASE}/agents/code/AGENTS.md.txt)

Extraia o ZIP numa pasta nova. Leia README.md e AGENTS.md. Confirme Python 3.10+ e espaço.
SHA-256 do ZIP: `{plan['package']['sha256']}`.

## 2. Instalar e executar

Na pasta raiz do projeto, execute:

```bash
{setup}
```

Windows: use `.venv\\Scripts\\Activate.ps1` no lugar de `source`.
O runner faz oito etapas em ordem, usa uma pasta isolada e interrompe se uma falhar.
Não instala dependências sozinho, não usa microfone, não chama APIs e não treina uma base gigante.

## 3. Conferir e documentar

Leia `work/talkweave-demo/REPORT.md`, `report.json` e `logs/`. O JSON registra comandos,
estados, tempos e SHA-256 das evidências. Confira também as métricas de treino e a
saída livre. Texto vazio, overfitting ou ruído devem aparecer no relatório; nunca
transforme execução bem-sucedida em uma afirmação de boa conversação.
Use `docs/report.md` como modelo para complementar o registro automático.
Retome com `--resume` somente quando as evidências forem conferidas pelo runner.

## 4. Próximo passo humano

O modelo inicial usa tons e não oferece conversação natural pronta. Depois do teste
sintético, prepare gravações humanas em outra pasta, siga o capítulo dados e avalie
pessoas fora do treino. O microfone real usa `python src/infer.py --microphone --ui`
após treinar checkpoints adequados. O console mostra dados reais do fluxo; não é
uma medição prévia da qualidade do modelo. O runner e a interface não implementam AEC.

## 5. Regras de código

{STYLE}

## Etapas verificáveis

'''+ '\n'.join(f"### {id}\n\n`python {cmd}`\n\nLeia o [capítulo {chapter}]({BASE}/#{chapter}). Espere os arquivos: {', '.join(files) or 'log de execução sem erro'}. Só avance após sucesso e evidência.\n" for id,cmd,files,chapter in steps)
    guide+='\n## Arquivos completos sem executar JavaScript\n\n'+'\n'.join(f"- [{r['path']}]({r['url']})" for r in records)
    (out/'guide.md').write_text(guide)
    (dist/'llms.txt').write_text(f'# TalkWeave\n\n> Tutorial de rede neural full-duplex em português, CPU primeiro.\n\n## Start\n- [Agent entry point]({BASE}/agents/): Readable HTML, no JavaScript required.\n- [Execution guide]({BASE}/agents/guide.md): Steps, commands, verification and documentation.\n- [Structured plan]({BASE}/agents/plan.json): Dependencies, files and hashes.\n- [Full reference]({BASE}/llms-full.txt): Guide plus complete tutorial.\n\nRespect the current site access policy. If login is required, use authorized access only.\n')
    (dist/'llms-full.txt').write_text(guide+'\n\n'+(dist/'downloads/talkweave-tutorial.md').read_text())
    cards=''.join(f'<li><strong>{html.escape(id)}</strong><code>python {html.escape(cmd)}</code><small>Evidência: {html.escape(", ".join(files) or "log e código de saída")}</small></li>' for id,cmd,files,_ in steps)
    page=f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Modo Agente — TalkWeave</title><meta name="description" content="Entrada legível para agentes de IA: comandos, etapas, protocolos e registro automático de evidências."><link rel="alternate" type="text/markdown" href="guide.md"><link rel="stylesheet" href="../style.css"><style>body{{padding:0}}.agent-wrap{{max-width:1000px;margin:auto;padding:42px 26px}}.agent-head{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:24px}}.agent-wrap h1{{font-size:48px;margin-top:40px}}.agent-wrap h2{{font-size:23px;margin-top:42px}}.agent-wrap p{{max-width:800px;color:var(--muted)}}.agent-wrap code{{font:12px/1.8 var(--mono);overflow-wrap:anywhere}}.agent-wrap pre{{white-space:pre-wrap;background:#1b2c28;color:#d2e3ca;padding:23px;border-radius:8px}}.agent-wrap ol{{padding:0;list-style:none;display:grid;gap:12px}}.agent-wrap li{{border:1px solid var(--line);padding:17px;border-radius:6px;background:white}}.agent-wrap li code,.agent-wrap li small{{display:block;margin-top:9px}}.agent-wrap li small{{color:var(--muted);font-size:11px}}.agent-links{{display:flex;gap:12px;flex-wrap:wrap}}.agent-links a{{font-size:12px;color:var(--teal);text-decoration:underline}}.agent-prompt{{width:100%;height:125px;resize:vertical;border:1px solid var(--line);border-radius:7px;padding:17px;font:13px/1.8 var(--sans);background:white;color:var(--ink)}}.tree{{font:12px/1.8 var(--mono)}}@media(max-width:600px){{.agent-wrap h1{{font-size:36px}}}}</style></head><body><main class="agent-wrap"><header class="agent-head"><a class="brand" href="../"><span class="brand-mark">∿</span><span>TALK<span class="brand-lab">WEAVE</span></span></a><a class="download" href="../downloads/talkweave.zip">↓ Projeto completo</a></header><div class="eyebrow" style="margin-top:40px">PARA AGENTES DE IA · SEM MCP · SEM API PAGA</div><h1>Um link. Um roteiro.<br><em>Evidências de verdade.</em></h1><p>Passe este link ao agente. Ele encontra os arquivos, executa oito etapas em CPU e registra comandos, resultados e limitações. Esta página funciona sem JavaScript.</p><div class="agent-links"><a href="guide.md">Guia em Markdown ↗</a><a href="plan.json">Plano JSON ↗</a><a href="../llms-full.txt">Referência completa ↗</a><a href="../#padroes">Padrões de código ↗</a></div><h2>Copie o pedido</h2><textarea class="agent-prompt" readonly aria-label="Pedido pronto para enviar ao agente">{html.escape(prompt)}</textarea><button class="btn-primary" id="copy-prompt" style="margin-top:12px">Copiar pedido</button><h2>1. Preparar e executar</h2><p>Extraia o ZIP numa pasta nova, leia README.md e AGENTS.md e execute na raiz:</p><pre><code>{html.escape(setup)}</code></pre><h2>2. Conferir cada etapa</h2><ol>{cards}</ol><h2>3. Entregar o registro</h2><p>O runner salva <code>work/talkweave-demo/REPORT.md</code>, <code>report.json</code> e <code>logs/</code>. Ele interrompe na falha e só retoma etapas com evidências intactas. O agente complementa a análise de qualidade usando <code>docs/report.md</code>.</p><h2>Uma organização pequena e explícita</h2><pre class="tree">run.py        → execução guiada\nsrc/          → áudio, modelo, protocolo e UI\ncfg/          → configurações\ndata/         → gravações locais\nruns/         → pesos e métricas\nlogs/         → evidências\ndocs/         → padrões e relatório\ntests/        → testes\nwork/         → experimentos isolados</pre><p>Funções com docstrings de uma linha em inglês, nomes claros, interfaces tipadas e filas limitadas. A interface é observadora do streaming, fora da callback de áudio.</p><div class="note warn"><span>△</span><div>Se o site solicitar login, use acesso autorizado ou o ZIP fornecido pelo usuário. O fixture usa tons: executar o pipeline não comprova reconhecimento de fala, português, AEC ou voz natural.</div></div><p><a href="guide.md">Leia o guia completo e a lista de arquivos sem JavaScript →</a></p></main><script>document.getElementById('copy-prompt').onclick=async function(){{try{{await navigator.clipboard.writeText(document.querySelector('textarea').value);this.textContent='Pedido copiado ✓';}}catch{{document.querySelector('textarea').select();this.textContent='Texto selecionado: copie manualmente';}}}};</script></body></html>'''
    (out/'index.html').write_text(page)
