"""Build sem dependências de frontend; conteúdo, arquivos e pacote reproduzível."""
import json
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path
from content import CHAPTERS, SOURCES
import agents
import ports

ROOT=Path(__file__).parent
PROJECT=ROOT/'lab'
DIST=ROOT/'docs'
DOWN=DIST/'downloads'
shutil.copytree(ROOT/'web', DIST, dirs_exist_ok=True)
DOWN.mkdir(exist_ok=True,parents=True)
agents.prepare(ROOT)
ports.prepare(ROOT)
files={str(p.relative_to(PROJECT)):p.read_text() for p in sorted(PROJECT.rglob('*')) if p.is_file() and p.suffix in {'.py','.txt','.md','.ini','.rs','.toml','.json'} and not any(part in {'work', 'runs', '__pycache__', '.venv'} for part in p.relative_to(PROJECT).parts)}
files['LICENSE']=(PROJECT/'LICENSE').read_text()
prior=json.loads((DOWN/'experiment-results.json').read_text()) if (DOWN/'experiment-results.json').exists() else {}
metrics=json.loads((PROJECT/'runs/duplex-metrics.json').read_text()) if (PROJECT/'runs/duplex-metrics.json').exists() else prior['duplex']
report=json.loads((PROJECT/'reply.wav.json').read_text()) if (PROJECT/'reply.wav.json').exists() else prior['free_generation']
verification='''Executado em CPU · Python 3.12 · PyTorch 2.15.0+cpu
9 testes passaram: envelope PCM, alinhamento, causalidade,
equivalência streaming/sequência, gradiente do codec, WER/shift,
contrato Frame/Result, docstrings de uma linha e dropout treino/inferência.
Codec: 100 passos · Duplex: 100 passos · seed 7
Fixture: 2 sessões de tons, sem voz humana.
Geração livre: WAV e relatório JSON produzidos.
Texto do usuário/agente vazio no fixture de validação.
Não verificado: microfone físico, GPU, precisão de fala humana.
Tempo por quadro registrado é apenas inferência local deste experimento,
não inclui captura, rede ou reprodução e não é promessa de desempenho.'''
payload=dict(chapters=CHAPTERS,sources=SOURCES,files=files,metrics=metrics,verification=verification)
(DIST/'data.js').write_text('const CONTENT = '+json.dumps(payload,ensure_ascii=False)+';\n')
for source,dest in [('data/validation-user.wav','validation-user.wav'),('reply.wav','reply.wav')]:
    if (PROJECT/source).exists(): shutil.copy2(PROJECT/source,DOWN/dest)
codec_metrics=json.loads((PROJECT/'runs/codec-metrics.json').read_text()) if (PROJECT/'runs/codec-metrics.json').exists() else prior['codec']
result={'fixture':'synthetic tones, no human speech','codec':codec_metrics,'duplex':metrics,'free_generation':report,'scope':verification}
(DOWN/'experiment-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))

# O manual offline preserva também o texto de cada laboratório e seus limites.
manual=['# TalkWeave — Da onda à conversa','Tutorial em português. Edição 01 — 8 de novembro de 2026.','Código original MIT; fontes externas conservam seus termos.']
for i,ch in enumerate(CHAPTERS):
    manual += ['\n## '+str(i+1).zfill(2)+' · '+ch['title'],ch['subtitle']]
    for block in ch['blocks']:
        t=block['type']
        if t=='h':manual.append('\n### '+block['text'])
        elif t in ('p','note'):manual.append(('> ' if t=='note' else '')+block['text'])
        elif t=='code':
            manual.append('```'+block['lang']+'\n'+block['text']+'\n```')
            if block.get('file'):manual.append('Arquivo completo: '+block['file'])
            if block.get('result'):manual.append('O que observar: '+block['result'])
        elif t=='table':
            manual.append('| '+' | '.join(block['headers'])+' |\n| '+' | '.join(['---']*len(block['headers']))+' |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in block['rows']))
        elif t=='lab':
            name=block['name']
            descriptions={
                'console':'Console Rich: captura de uma saída real sobre tons sintéticos. Rode python run.py --ui ou python src/infer.py --microphone --ui para a interface local.',
                'architecture':'Áudio usuário → codec → embeddings → GRU causal. História do agente → embeddings → GRU. Saídas: áudio do agente, texto usuário, texto agente, eventos. A saída anterior realimenta a rede.',
                'signal':'Experimento interativo no site: senoide+ruído, RMS/ZCR calculados. O script baseline.py contém atributos equivalentes. Não identifica fala.',
                'logistic':'Experimento interativo no site: descida de gradiente real sobre dados sintéticos. Execute baseline.py para a versão NumPy.',
                'wer':'Experimento interativo no site: distância de edição entre textos. core.wer fornece versão Python.',
                'loss':json.dumps(metrics,ensure_ascii=False),
                'artifacts':'Ouça downloads/validation-user.wav e downloads/reply.wav no site. Ambos contêm tons/saída sintética, sem fala humana.',
                'timeline':'Simulação visual: captura permanece ativa enquanto agente e worker operam; interrupção invalida saída antiga. Não é inferência real.',
                'budget':'Pesos FP32=4B/parâmetro; com gradientes+Adam ≈16B/parâmetro antes de ativações. WAV estéreo16k PCM16 = 230.4MB/h em unidade decimal.',
                'files':'Arquivos completos estão na pasta deste projeto e no visualizador do site.',
                'verification':verification,
                'sources':'Veja as referências ao final.'}
            manual.append(descriptions[name])
    manual.append('\nAntes de seguir:\n'+'\n'.join('- [ ] '+s for s in ch['checks']))
    for id in ch['sources']:
        s=next(s for s in SOURCES if s[0]==id);manual.append(f'Fonte: [{s[1]}]({s[2]})')
manual+=['\n## Referências']+[f'- [{s[1]}]({s[2]}) — {s[3]}' for s in SOURCES]
text='\n\n'.join(manual)+'\n'
(DOWN/'talkweave-tutorial.md').write_text(text)
(PROJECT/'TUTORIAL.md').write_text(text)
with zipfile.ZipFile(DOWN/'talkweave.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PROJECT.rglob('*')):
        if any(part in {'work', '.venv'} for part in p.relative_to(PROJECT).parts):continue
        if not p.is_file() or any(part in {'__pycache__','.pytest_cache','runs'} for part in p.parts) or p.suffix in {'.wav','.npz'} or p.name=='reply.wav.json':continue
        z.write(p,'talkweave/'+str(p.relative_to(PROJECT)))
    z.write(DOWN/'experiment-results.json','talkweave/experiment-results.json')
print(json.dumps({'chapters':len(CHAPTERS),'words':len(text.split()),'files':len(files),'zip_bytes':(DOWN/'talkweave.zip').stat().st_size},ensure_ascii=False))

agents.publish(ROOT, CHAPTERS)

ports.publish(ROOT)
