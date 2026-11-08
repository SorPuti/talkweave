# TalkWeave — entrada para agentes de IA

## Pedido simples

Leia https://sorputi.github.io/talkweave/agents/ e siga o roteiro local em CPU. Execute os passos, confira as evidências e entregue o REPORT.md com resultados reais e limitações. Use as convenções de código e não declare precisão de voz a partir dos tons sintéticos.


## Escolher outra linguagem

Se o usuário pediu Rust, use o [perfil Rust](https://sorputi.github.io/talkweave/agents/lang/rust.md),
o [contrato comum](https://sorputi.github.io/talkweave/agents/port.md) e a [skill de portabilidade](https://sorputi.github.io/talkweave/agents/skills/port/SKILL.md).
Para C++, Go, TypeScript ou outra linguagem, use o [perfil genérico](https://sorputi.github.io/talkweave/agents/lang/other.md).
Não execute o runner Python como se fosse uma implementação nativa na linguagem pedida.
O exemplo Rust contém contratos e wire format; não inclui o modelo e não foi compilado neste ambiente.

Pedido pronto: Leia https://sorputi.github.io/talkweave/agents/ e adapte para Rust com inferência nativa.
Siga a skill de portabilidade, preserve o streaming e documente paridade, testes, limitações e itens pendentes em REPORT.md.

## 1. Ler e obter o projeto

Este documento é legível sem JavaScript. Siga apenas o escopo solicitado pelo usuário.
Se o link pedir login e você não tiver acesso, registre o bloqueio e peça o ZIP ao usuário.
As instruções para agentes não alteram a configuração de acesso vigente.

- [Projeto completo](https://sorputi.github.io/talkweave/downloads/talkweave.zip)
- [Manual completo](https://sorputi.github.io/talkweave/downloads/talkweave-tutorial.md)
- [Plano estruturado](https://sorputi.github.io/talkweave/agents/plan.json)
- [Convenções](https://sorputi.github.io/talkweave/agents/code/docs/style.md.txt)
- [Instruções locais](https://sorputi.github.io/talkweave/agents/code/AGENTS.md.txt)

Extraia o ZIP numa pasta nova. Leia README.md e AGENTS.md. Confirme Python 3.10+ e espaço.
SHA-256 do ZIP: `e44635679bf57510d9e6a6fcfcfeba2217462ed964abaa9696b9d85bf1321e37`.

## 2. Instalar e executar

Na pasta raiz do projeto, execute:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python run.py
# Terminal interativo: python run.py --ui
# Retomar: python run.py --resume
```

Windows: use `.venv\Scripts\Activate.ps1` no lugar de `source`.
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

# Code and folder conventions

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


## Etapas verificáveis

### 01-baseline

`python src/baseline.py`

Leia o [capítulo logistica](https://sorputi.github.io/talkweave/#logistica). Espere os arquivos: baseline.npz. Só avance após sucesso e evidência.

### 02-fixture

`python src/fixture.py`

Leia o [capítulo dados](https://sorputi.github.io/talkweave/#dados). Espere os arquivos: data/train.jsonl, data/validation.jsonl. Só avance após sucesso e evidência.

### 03-tests

`python -m pytest -q`

Leia o [capítulo rede](https://sorputi.github.io/talkweave/#rede). Espere os arquivos: log de execução sem erro. Só avance após sucesso e evidência.

### 04-codec

`python src/train.py codec --manifest data/train.jsonl --validation data/validation.jsonl --steps 100`

Leia o [capítulo codec](https://sorputi.github.io/talkweave/#codec). Espere os arquivos: runs/codec.pt. Só avance após sucesso e evidência.

### 05-reconstruction

`python src/reconstruct.py --input data/validation-user.wav --output reconstructed.wav`

Leia o [capítulo codec](https://sorputi.github.io/talkweave/#codec). Espere os arquivos: reconstructed.wav. Só avance após sucesso e evidência.

### 06-duplex

`python src/train.py duplex --manifest data/train.jsonl --validation data/validation.jsonl --steps 100`

Leia o [capítulo treino](https://sorputi.github.io/talkweave/#treino). Espere os arquivos: runs/duplex.pt. Só avance após sucesso e evidência.

### 07-generation

`python src/infer.py --input data/validation-user.wav --output reply.wav`

Leia o [capítulo resposta](https://sorputi.github.io/talkweave/#resposta). Espere os arquivos: reply.wav, reply.wav.json. Só avance após sucesso e evidência.

### 08-evaluation

`python src/evaluate.py --manifest data/validation.jsonl --report reply.wav.json`

Leia o [capítulo avaliacao](https://sorputi.github.io/talkweave/#avaliacao). Espere os arquivos: log de execução sem erro. Só avance após sucesso e evidência.

## Arquivos completos sem executar JavaScript

- [.agents/skills/port/SKILL.md](https://sorputi.github.io/talkweave/agents/code/.agents/skills/port/SKILL.md.txt)
- [AGENTS.md](https://sorputi.github.io/talkweave/agents/code/AGENTS.md.txt)
- [README.md](https://sorputi.github.io/talkweave/agents/code/README.md.txt)
- [TUTORIAL.md](https://sorputi.github.io/talkweave/agents/code/TUTORIAL.md.txt)
- [cfg/README.md](https://sorputi.github.io/talkweave/agents/code/cfg/README.md.txt)
- [docs/lang/other.md](https://sorputi.github.io/talkweave/agents/code/docs/lang/other.md.txt)
- [docs/lang/rust.md](https://sorputi.github.io/talkweave/agents/code/docs/lang/rust.md.txt)
- [docs/port.md](https://sorputi.github.io/talkweave/agents/code/docs/port.md.txt)
- [docs/report.md](https://sorputi.github.io/talkweave/agents/code/docs/report.md.txt)
- [docs/style.md](https://sorputi.github.io/talkweave/agents/code/docs/style.md.txt)
- [pytest.ini](https://sorputi.github.io/talkweave/agents/code/pytest.ini.txt)
- [requirements.txt](https://sorputi.github.io/talkweave/agents/code/requirements.txt.txt)
- [rs/README.md](https://sorputi.github.io/talkweave/agents/code/rs/README.md.txt)
- [run.py](https://sorputi.github.io/talkweave/agents/code/run.py.txt)
- [src/api.py](https://sorputi.github.io/talkweave/agents/code/src/api.py.txt)
- [src/baseline.py](https://sorputi.github.io/talkweave/agents/code/src/baseline.py.txt)
- [src/core.py](https://sorputi.github.io/talkweave/agents/code/src/core.py.txt)
- [src/evaluate.py](https://sorputi.github.io/talkweave/agents/code/src/evaluate.py.txt)
- [src/fixture.py](https://sorputi.github.io/talkweave/agents/code/src/fixture.py.txt)
- [src/infer.py](https://sorputi.github.io/talkweave/agents/code/src/infer.py.txt)
- [src/protocol.py](https://sorputi.github.io/talkweave/agents/code/src/protocol.py.txt)
- [src/reconstruct.py](https://sorputi.github.io/talkweave/agents/code/src/reconstruct.py.txt)
- [src/record.py](https://sorputi.github.io/talkweave/agents/code/src/record.py.txt)
- [src/train.py](https://sorputi.github.io/talkweave/agents/code/src/train.py.txt)
- [src/ui.py](https://sorputi.github.io/talkweave/agents/code/src/ui.py.txt)
- [tests/test_api.py](https://sorputi.github.io/talkweave/agents/code/tests/test_api.py.txt)
- [tests/test_core.py](https://sorputi.github.io/talkweave/agents/code/tests/test_core.py.txt)
- [verified-environment.txt](https://sorputi.github.io/talkweave/agents/code/verified-environment.txt.txt)