"""Publish language-neutral adaptation rules and a small Rust contract starter."""
import hashlib
import html
import json
import struct
from pathlib import Path

BASE='https://sorputi.github.io/talkweave'
SKILL='''---
name: duplex-port
description: Adapt the TalkWeave tutorial to a requested programming language while preserving streaming behavior and verifiable evidence.
---

# Adapt the tutorial to another language

Apply this guide only when the user requests a language adaptation. Respect the
user's instructions and the agent's own policies. This is a downloadable skill
file, not an automatically installed plugin or a grant of additional permissions.

1. Record the target language, compiler/runtime, hardware, and requested scope in
   docs/port.md. Use the language the user named; do not silently default to Python.
2. Read AGENTS.md, docs/style.md, docs/port.md and docs/lang/<profile>.md. If no profile
   exists, use docs/lang/other.md. Read the reference Python files for behavior.
3. Choose explicitly: hybrid runtime with Python model, native inference with
   existing weights, or native training plus inference. Explain that a Python
   subprocess/FFI bridge is hybrid, not a native model implementation.
4. Keep the boundaries: Frame, Result, Engine, transport, presentation. Use short
   folders src/, cfg/, data/, runs/, logs/, docs/, tests/, work/. Preserve units,
   sample rate, frame clock, codec version, codebook order, text vocabulary,
   recurrent state, blank semantics, and previous-agent feedback.
5. Document public functions in one English line using the language's convention:
   Rust ///, C++ /// or /** ... */, Go // Name ..., TypeScript /** ... */. Use the
   language's formatter and type system; do not imitate Python syntax in Rust.
6. Port and test PCM framing first using cfg/wire.json. Then compare audio framing,
   codec codes, recurrent state, logits, generated audio, and text to the reference.
   Integer wire bytes must match exactly. Select and record floating tolerances
   for each backend; do not assume one tolerance is correct for every precision.
7. Test whole-sequence versus frame-by-frame inference, future-input mutation,
   session reset, frame gaps, repeated frames, invalid rates, nonfinite samples,
   bounded queues, and interruption/cancellation with stale output.
8. Keep allocation-heavy work, inference, tools, logs and animated UI outside the
   capture callback. Never feed slow console rendering back into audio scheduling.
   Measure p50/p95, queues, underruns, RTF and task quality; native code is not an
   automatic speed or precision improvement.
9. Save exact commands, exit codes, compiler/dependency versions and artifacts in
   REPORT.md, report.json and logs/. Mark steps passed, failed, blocked or untested.
   Separate build success, numerical parity, performance and human voice quality.
10. If a dependency/backend is unsupported, record it and the next concrete step.
    Never replace the requested neural engine with a mock and claim completion.

Completion requires: build/tests pass on the actual target, parity tests pass,
streaming is exercised, limitations are documented, and the report names the
chosen hybrid/native scope. Training is included only when actually implemented.
'''
PORT='''# Adaptação por linguagem

O tutorial não depende de Python como conceito. O pacote de referência e o runner
atual são Python; uma portabilidade precisa preservar os contratos e validar a
implementação escolhida. O site fornece instruções para o agente, não um conversor
que transforma automaticamente a rede em qualquer linguagem.

## Escolha o escopo antes do código

| Escopo | O que muda | O que não pode ser afirmado |
|---|---|---|
| Híbrido | Captura/UI/filas na linguagem escolhida; modelo Python via processo ou FFI | Não é um modelo neural totalmente nativo |
| Inferência nativa | Codec, GRU, cabeças, estado e decodificação na linguagem alvo | Não inclui treinamento nativo só por carregar pesos |
| Tudo nativo | Dados, gradientes, perdas, otimizador, checkpoints e inferência | Exige validar treino e inferência, não só compilar |

## Ordem de trabalho

1. Confirme a linguagem pedida, o compilador e o escopo; registre em docs/port.md.
2. Defina Frame, Result e Engine com as unidades da referência.
3. Implemente e teste o wire format com cfg/wire.json.
4. Compare quadro e reamostragem; nunca mude a taxa só no cabeçalho.
5. Compare encoder, tabela VQ, decoder, embeddings, GRU e cabeças com pesos idênticos.
6. Preserve o estado entre quadros e realimente áudio/texto previstos na inferência.
7. Valide causalidade, reset de sessão, rejeição de replay e geração livre.
8. Conecte captura/reprodução e depois uma UI fora da callback.
9. Meça velocidade e qualidade separadas e escreva REPORT.md com evidências.

## Contrato compartilhado

- Entrada neural: mono float32, 16 kHz, 320 amostras, sequência crescente por sessão.
- Saída: áudio do agente, texto do usuário, texto do agente, cinco escores de eventos,
  tempo medido por quadro. Escores não são confiança calibrada.
- Modelo: codec com tabela de 256 códigos; GRU causal; vocabulário CHARS da referência.
- Estado: persistente durante a sessão, reiniciado somente pela política documentada.
- Wire: `<BQQH` little-endian, 19 bytes, versão 1, seq u64, sample_clock u64, n u16,
  seguido por n amostras PCM16LE. O envelope permite tamanhos variáveis; o engine
  deste laboratório aceita somente os 320 valores de seu contrato.
- Quantização PCM: limitar a [-1, 1], multiplicar por 32767 em float32 e arredondar
  ties-to-even. Rust round() comum não replica o arredondamento NumPy nos empates.

## O que comparar

Use os mesmos pesos e a mesma sequência de entrada. Registre diferenças em códigos
VQ, h[t], logits de cada cabeça e áudio decodificado. Antes de comparar operações,
confira layout dos pesos, ordem dos gates GRU e convenção das biases. Um runtime
com ordem de gates diferente pode compilar e produzir uma rede diferente.
ONNX ou outro formato de exportação é uma opção de engenharia, não compatibilidade
já comprovada: valide operadores, entradas/saídas de estado e formatos/dtypes.

Bytes e índices inteiros devem ser iguais. Tolerâncias de ponto flutuante dependem
do backend e precisão e devem ser justificadas no relatório. Argmax pode mudar com
pequenas diferenças em logits próximos; examine a diferença, não esconda o erro.
Se portar o treino, compare também perdas, gradientes e um passo do otimizador em
um minibatch fixo antes de um treinamento longo.

## Experiência de console

O backend não conhece o terminal. Exponha um snapshot de métricas com filas limitadas
e renderize em outra tarefa, a até oito atualizações/s como ponto inicial. Use uma UI
idiomática da linguagem e saída JSON/texto quando não houver TTY. Dê destaque a erros,
etapa atual, duração e arquivos gerados. Não gere transcrições ou percentuais fictícios.

## Relatório de conclusão

Informe linguagem, toolchain, escopo escolhido, comandos e versões, testes de wire,
paridade numérica, causalidade, streaming, medições, falhas e itens não testados.
Use docs/report.md. O projeto Rust incluído é um início de contrato/protocolo,
não contém rede, captura, treinamento nem UI nativos prontos.
'''
RUST='''# Perfil Rust

Leia ../port.md e .agents/skills/port/SKILL.md. O exemplo em rs/ compila apenas
contratos e serialização sem crates externas. É um ponto inicial, não a rede portada.
Nesta edição não havia rustc/cargo no ambiente de validação: compilação Rust e seus
testes ficam explicitamente não verificados até o agente executá-los na máquina alvo.

## Começar

```bash
rustc --version
cargo --version
cd rs
cargo fmt
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```

O Cargo.toml exige Rust >=1.77 para round_ties_even. Registre a versão realmente
usada. Não instale toolchains ou modifique um ambiente compartilhado sem respeitar
a autorização do usuário. A amostra não implementa capture/device permissions.

## Organização e estilo

Use Cargo, src/lib.rs para contratos e tests/ para testes adicionais. Funções em
snake_case, tipos/traits em PascalCase, constantes em SCREAMING_SNAKE_CASE. Documente
itens públicos com /// e uma frase em inglês; use Result para falhas e evite panic
na entrada externa. Prefira buffers pré-alocados e uma fila limitada entre callback,
inferência e terminal. Async é útil para coordenação; não cria um backend neural.

## Interfaces

Frame contém sequência e [f32; 320]. Output contém áudio gerado, os dois textos,
escores de eventos e elapsed_ms. Engine::process(&mut self, frame: &Frame) preserva
estado por sessão. Não deixe a UI compartilhar mutavelmente o estado da GRU.

## Caminho até a rede nativa

1. Execute os testes de wire e compare cfg/wire.json com o Python.
2. Escolha um backend de tensores/inferência com os operadores necessários e confira
   sua documentação atual. Não assuma que ele suporta o treino ou export de GRU.
3. Exporte pesos de forma versionada e documente os nomes, shapes, dtype, layout de
   gates e biases. Nunca carregue um checkpoint diferente sem identificá-lo.
4. Porte codec e compare códigos/reconstruções. Porte GRU e compare estado/logits.
5. Implemente geração livre com o próprio código/texto anterior do agente.
6. Só depois adicione captura, AEC se necessário e console. Primeiro teste com fones.
7. Meça no mesmo computador e entrada. Rust não garante mais precisão: a fidelidade
   vem da rede, dos dados e da preservação das operações.

## Opção híbrida

Você pode deixar o modelo Python em um worker local e fazer o runtime em Rust.
Documente IPC, ordem e tamanho das mensagens, backpressure, timeout, cancelamento
por turno e encerramento de sessão. Isso pode ser útil, mas continua híbrido.
Não chame roundtrip por processo novo a cada frame: mantenha sessão/processo vivos.
'''
OTHER='''# Perfil para outra linguagem

Use docs/port.md e a skill duplex-port. Informe a linguagem solicitada no relatório.
Não escolha Python por padrão se o usuário pediu outra. Preserve contratos e testes.

| Linguagem | Convenção de documentação | Checagem inicial sugerida |
|---|---|---|
| C++ | /// ou /** frase em inglês */ em funções públicas | toolchain, build e testes do projeto; sanitizers quando apropriado |
| Go | // NomeDaFuncao + frase em inglês | gofmt e go test ./... |
| TypeScript | /** frase em inglês */ com tipos explícitos | tsc --noEmit e testes do projeto |
| Outra | Comentário de documentação idiomático, uma linha em inglês | formatter, compilador/type checker e testes equivalentes |

Estas são orientações, não backends já integrados. Valide bibliotecas atuais no alvo,
operadores neurais, suporte a áudio, relógio, estado e thread de callback. TypeScript
em Node e no navegador têm APIs e permissões diferentes. Go ou C++ também precisam
de um backend neural real; uma interface vazia não implementa uma rede.

Comece por wire bytes, depois framing, codec, estado, logits, geração livre e captura.
Se não houver suporte ao treino nativo, escolha e documente inferência ou híbrido.
Entregue REPORT.md com evidências, limitações e componentes ainda não implementados.
'''
CARGO='''[package]
name = "duplex-port"
version = "0.1.0"
edition = "2021"
rust-version = "1.77"

[lib]
path = "src/lib.rs"
'''
LIB=r'''//! Define streaming contracts and the v1 PCM wire format; no neural model is included.

pub const RATE: u32 = 16_000;
pub const SIZE: usize = 320;
pub const HEAD: usize = 19;

/// Carry one audio frame on a session sequence.
pub struct Frame {
    pub seq: u64,
    pub samples: [f32; SIZE],
}

/// Carry generated audio, text, model scores, and measured processing time.
pub struct Output {
    pub audio: [f32; SIZE],
    pub user: String,
    pub agent: String,
    pub events: [f32; 5],
    pub elapsed_ms: f64,
}

/// Specify a stateful causal engine without implementing a neural backend.
pub trait Engine {
    /// Consume one current frame while preserving the session state.
    fn process(&mut self, frame: &Frame) -> Result<Output, &'static str>;
}

/// Encode finite PCM samples with the Python reference rounding convention.
pub fn pack(seq: u64, clock: u64, samples: &[f32]) -> Result<Vec<u8>, &'static str> {
    if samples.len() > u16::MAX as usize || samples.iter().any(|x| !x.is_finite()) {
        return Err("invalid samples");
    }
    let mut out = Vec::with_capacity(HEAD + samples.len() * 2);
    out.push(1);
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&clock.to_le_bytes());
    out.extend_from_slice(&(samples.len() as u16).to_le_bytes());
    for x in samples {
        let pcm = (x.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16;
        out.extend_from_slice(&pcm.to_le_bytes());
    }
    Ok(out)
}

/// Represent a decoded wire sequence, sample clock, and PCM samples.
pub type Packet = (u64, u64, Vec<f32>);

/// Decode a validated packet without imposing the neural engine frame size.
pub fn unpack(packet: &[u8]) -> Result<Packet, &'static str> {
    if packet.len() < HEAD || packet[0] != 1 {
        return Err("invalid header");
    }
    let mut seq = [0_u8; 8];
    seq.copy_from_slice(&packet[1..9]);
    let mut clock = [0_u8; 8];
    clock.copy_from_slice(&packet[9..17]);
    let n = u16::from_le_bytes([packet[17], packet[18]]) as usize;
    if packet.len() != HEAD + n * 2 {
        return Err("invalid length");
    }
    let audio = packet[HEAD..].chunks_exact(2)
        .map(|x| i16::from_le_bytes([x[0], x[1]]) as f32 / 32767.0)
        .collect();
    Ok((u64::from_le_bytes(seq), u64::from_le_bytes(clock), audio))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Match fixed Python bytes including ties-to-even PCM rounding.
    #[test]
    fn golden_bytes() {
        let packet = pack(7, 960, &[-1.0, -0.5, 0.0, 0.5, 1.0]).unwrap();
        assert_eq!(packet, vec![1,7,0,0,0,0,0,0,0,192,3,0,0,0,0,0,0,5,0,1,128,0,192,0,0,0,64,255,127]);
        let (seq, clock, audio) = unpack(&packet).unwrap();
        assert_eq!((seq, clock), (7, 960));
        assert_eq!(audio.len(), 5);
        assert_eq!(audio[0], -1.0);
    }

    /// Reject truncation and nonfinite sample values.
    #[test]
    fn invalid_input() {
        assert!(unpack(&[1, 0]).is_err());
        assert!(pack(0, 0, &[f32::NAN]).is_err());
        let mut packet = pack(0, 0, &[0.0]).unwrap();
        packet.pop();
        assert!(unpack(&packet).is_err());
    }
}
'''

def prepare(root):
    project=root/'lab'
    docs=project/'docs'
    (docs/'lang').mkdir(exist_ok=True,parents=True)
    skill=project/'.agents/skills/port';skill.mkdir(parents=True,exist_ok=True)
    (skill/'SKILL.md').write_text(SKILL)
    (docs/'port.md').write_text(PORT)
    (docs/'lang/rust.md').write_text(RUST)
    (docs/'lang/other.md').write_text(OTHER)
    rust=project/'rs';(rust/'src').mkdir(parents=True,exist_ok=True)
    (rust/'Cargo.toml').write_text(CARGO);(rust/'src/lib.rs').write_text(LIB)
    (rust/'README.md').write_text('# Rust starter\n\nContracts and wire format only; no neural implementation. See ../docs/lang/rust.md. Rust compiler tests are unverified in the build environment.\n')
    values=[-1.,-.5,0.,.5,1.]
    packet=struct.pack('<BQQH',1,7,960,len(values))+struct.pack('<5h',-32767,-16384,0,16384,32767)
    (project/'cfg/wire.json').write_text(json.dumps({'version':1,'seq':7,'sample_clock':960,'samples':values,'pcm16':[-32767,-16384,0,16384,32767],'hex':packet.hex(),'rounding':'float32, ties-to-even','header_bytes':19},indent=2))
    path=project/'AGENTS.md'
    path.write_text(path.read_text().replace('# TalkWeave — agent instructions', '# TalkWeave — agent instructions\n\nIf the user named another language, read docs/port.md and the matching language profile first. The Python execution steps below apply only to the Python reference.\n')+'''\n## Requested language takes precedence\n\nThe bundled runner is Python. If the user requests Rust or another language, read\ndocs/port.md, docs/lang/rust.md or docs/lang/other.md, and\n.agents/skills/port/SKILL.md before coding. Record hybrid/native inference/native\ntraining scope. Preserve contracts and run parity tests. A Python worker is hybrid,\nnot a completed native port. Rust source in rs/ is only a protocol/trait starter;\nno compiler test or neural backend is claimed.\n''')

def publish(root):
    dist=root/'docs';out=dist/'agents';lang=out/'lang';lang.mkdir(exist_ok=True)
    for name,text in [('rust',RUST),('other',OTHER)]:
        links=f'\n[Contrato comum]({BASE}/agents/port.md) · [Skill]({BASE}/agents/skills/port/SKILL.md) · [Projeto completo]({BASE}/downloads/talkweave.zip)\n\n'
        (lang/(name+'.md')).write_text(links+text)
    (out/'port.md').write_text(PORT)
    skill=out/'skills/port';skill.mkdir(parents=True,exist_ok=True);(skill/'SKILL.md').write_text(SKILL)
    profiles={'python':{'status':'reference-tested-cpu','entry':BASE+'/agents/guide.md','run':'python run.py'},
              'rust':{'status':'adaptation-guide-and-wire-starter; compiler-unverified; no neural backend','entry':BASE+'/agents/lang/rust.md','starter':'rs/','checks':['cargo fmt --check','cargo test','cargo clippy --all-targets -- -D warnings']},
              'other':{'status':'language-neutral-adaptation-guide','entry':BASE+'/agents/lang/other.md'}}
    plan_path=out/'plan.json';plan=json.loads(plan_path.read_text());plan['language_profiles']=profiles
    plan['port_skill']=BASE+'/agents/skills/port/SKILL.md'
    for p in [root/'lab/rs/Cargo.toml',root/'lab/rs/src/lib.rs',root/'lab/cfg/wire.json']:
        rel=p.relative_to(root/'lab');target=out/'code'/Path(str(rel)+'.txt');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(p.read_text())
        if not any(f['path']==str(rel) for f in plan['files']):plan['files'].append({'path':str(rel),'url':BASE+'/agents/code/'+str(rel)+'.txt','sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    intro=f'''\n## Escolher outra linguagem\n\nSe o usuário pediu Rust, use o [perfil Rust]({BASE}/agents/lang/rust.md),\no [contrato comum]({BASE}/agents/port.md) e a [skill de portabilidade]({BASE}/agents/skills/port/SKILL.md).\nPara C++, Go, TypeScript ou outra linguagem, use o [perfil genérico]({BASE}/agents/lang/other.md).\nNão execute o runner Python como se fosse uma implementação nativa na linguagem pedida.\nO exemplo Rust contém contratos e wire format; não inclui o modelo e não foi compilado neste ambiente.\n\nPedido pronto: Leia {BASE}/agents/ e adapte para Rust com inferência nativa.\nSiga a skill de portabilidade, preserve o streaming e documente paridade, testes, limitações e itens pendentes em REPORT.md.\n'''
    guide=out/'guide.md';guide.write_text(guide.read_text().replace('## 1. Ler e obter o projeto', intro+'\n## 1. Ler e obter o projeto'))
    (dist/'llms.txt').write_text((dist/'llms.txt').read_text()+f'\n## Other languages\n- [Portability rules]({BASE}/agents/port.md)\n- [Rust profile]({BASE}/agents/lang/rust.md)\n- [Other language profile]({BASE}/agents/lang/other.md)\n- [Agent skill]({BASE}/agents/skills/port/SKILL.md)\n')
    (dist/'llms-full.txt').write_text((dist/'llms-full.txt').read_text()+intro+'\n\n'+PORT+'\n\n'+RUST+'\n\n'+OTHER+'\n\n'+SKILL)
    index=out/'index.html';page=index.read_text()
    card=f'''<section><h2>Quer construir em outra linguagem?</h2><p>Escolha a linguagem no pedido. O agente deve seguir os contratos comuns e declarar se entrega uma integração híbrida, inferência nativa ou treino nativo.</p><div class="agent-links"><a href="lang/rust.md">Rust: roteiro e starter ↗</a><a href="lang/other.md">C++ · Go · TypeScript · outras ↗</a><a href="port.md">Contratos e paridade ↗</a><a href="skills/port/SKILL.md">Skill para o agente ↗</a></div><pre><code>Leia {BASE}/agents/ e adapte para Rust com inferência nativa.\nSiga a skill de portabilidade e entregue REPORT.md com evidências.</code></pre><p>Rust: contratos e serialização iniciais em <code>rs/</code>. O modelo neural ainda precisa ser implementado; o starter não foi compilado neste ambiente. A skill é um arquivo para o agente ler, sem instalação automática.</p></section>'''
    page=page.replace('<h2>1. Preparar e executar</h2>',card+'<h2>1. Preparar e executar a referência Python</h2>')
    index.write_text(page)

PORT += "\n## Dropout e paridade\n\nA referência aceita dropout explícito de entrada e saída da GRU no treino. Na inferência, preserve o equivalente a eval(): nenhuma máscara aleatória. Não confunda dropout entre camadas de uma GRU com dropout recorrente; a referência tem uma camada. Teste paridade em modo de inferência e documente diferenças de regularização se portar o treino.\n"
