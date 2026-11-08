# TalkWeave — laboratório executável

Implementação didática original, MIT. CPU, Python 3.10+. Não acompanha modelo de conversa treinado. Os tons do fixture não são fala. O codec VQ de um único codebook é propositalmente pequeno e tem fidelidade limitada.

## Instalar e executar

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python src/baseline.py
python src/fixture.py
python -m pytest -q
python src/train.py codec --manifest data/train.jsonl --validation data/validation.jsonl --steps 100
python src/train.py duplex --manifest data/train.jsonl --validation data/validation.jsonl --steps 100
python src/infer.py --input data/validation-user.wav --output reply.wav
python src/evaluate.py --manifest data/validation.jsonl --report reply.wav.json
```

Teste de condicionamento textual experimental: `python src/infer.py --text 'olá mundo' --seconds 3 --output text.wav`. Com pesos de tons, a saída será tom/ruído, não voz. Para voz precisa treinar codec e duplex com gravações humanas.

## Gravações reais

Cada linha JSONL aponta para dois WAVs mono 16 kHz, mesmo número de amostras e mesma origem temporal. Não retire os silêncios. Arquivos relativos à pasta do manifesto. Use uma sessão por exemplo de até 6 segundos nesta versão (o treinador usa os primeiros 300 quadros). Recorte sessões longas em janelas antes do treino, preservando prefixo e rebaseando anotações. Separe locutores e sessões entre treino/validação/teste; liste ambos os locutores em `speakers`.

```json
{"session":"001","speakers":["ana","bia"],"language":"pt","user":"001-user.wav","agent":"001-agent.wav","user_words":[{"start":0.2,"end":0.6,"text":"olá "}],"agent_words":[{"start":0.9,"end":1.4,"text":"oi "}],"events":[{"start":0.2,"end":0.6,"label":"speech"},{"start":1.6,"end":1.9,"label":"laughter"}]}
```

Os eventos se referem ao usuário; são multi-label. Use espaços ao final das palavras para reconstruir texto. Inclua risadas genuínas, fala sobreposta, ruídos, silêncio, sotaques e microfones diferentes. Alinhamento manual em Audacity serve para o primeiro conjunto; marcação errada ensina erro. Nunca rotule uma risada como palavra. O código não faz diarização nem separa duas pessoas misturadas no mesmo canal.

## Microfone local

```bash
python -m pip install sounddevice
python -m sounddevice
python src/infer.py --microphone
```

Linux pode exigir `libportaudio2` pelo gerenciador de pacotes. Use fones: esse runner não implementa AEC. Ctrl+C encerra e imprime contadores de fila. Perdas/underruns invalidam medições de latência. O callback somente enfileira áudio; inferência fica fora dele. Você ouve saída do modelo, inclusive ruído se mal treinado: comece com volume baixo.

## O que foi implementado

Logística NumPy, codec VQ treinável com straight-through, GRU causal com cabeças de áudio/texto do usuário/texto do agente/eventos, treino supervisionado, validação, geração autoregressiva livre, áudio/texto condicional experimental, captura e reprodução simultâneas, envelope PCM16 e testes de causalidade. Não há LLM de conhecimento geral, reconhecimento universal, AEC, WebSocket server ou agente de navegador prontos neste pacote.

Treino usa teacher forcing (áudio e texto anteriores verdadeiros); inferência realimenta as próprias previsões. A diferença gera exposure bias: valide geração livre e preserve um teste que não entrou em nenhuma decisão de treino. Não confunda loss baixa com voz boa. Eventos usam F1 por quadro; avalie eventos também com tolerância temporal em gravações humanas. WER não mede timbre, naturalidade ou interrupção.

## Escalar com critério

O pacote carrega dados na RAM e limita sequências a 6s para facilitar execução em CPU. Não é um pipeline para centenas de horas. Próximos passos: dados em shards, RVQ causal com múltiplos codebooks, perdas STFT multiescala, codec congelado validado em PT/EN, tokenização subword, treino em chunks com estado e desligamento de gradiente, GRU maior/transformer causal com cache, scheduled sampling cuidadosamente avaliado. Mudar codec exige retokenizar todos os dados e retreinar o modelo de diálogo.

## Licença

MIT para código original deste laboratório. Pesos, dados e projetos citados no tutorial conservam suas próprias licenças.

## Execução guiada e agentes

Leia AGENTS.md e docs/style.md. Depois da instalação, rode `python run.py` para executar oito etapas em uma pasta isolada e gerar REPORT.md, report.json e logs. Use `python run.py --ui` para console animado no terminal ou `python run.py --resume` para retomar com hashes conferidos.

Implementação em src/, convenções em docs/, configuração em cfg/, testes em tests/ e experimentos em work/. Funções têm docstrings de uma linha em inglês. O contrato Frame/Result/Engine fica em src/api.py. Microfone: `python src/infer.py --microphone --ui`. Sem terminal interativo, a saída continua em texto simples.

## Rust e outras linguagens

Leia docs/port.md e docs/lang/rust.md ou docs/lang/other.md. A skill .agents/skills/port/SKILL.md define o roteiro para o agente. rs/ contém apenas um starter de contratos e serialização, sem rede neural ou treino nativos. O compilador Rust não estava disponível no ambiente de verificação. Preserve os bytes de cfg/wire.json e valide paridade de pesos, estados, logits e streaming antes de declarar uma portabilidade concluída.

## Dropout opcional

O estágio duplex aceita `--dropout 0.1`; o padrão continua 0 para preservar a referência. Aplica-se aos embeddings de entrada e à saída da GRU, não diretamente ao estado recorrente. Use `model.train()` no treino e `model.eval()` na inferência. `no_grad()` sozinho não desliga dropout. O treinador salva duplex-config.json. Compare 0/0.1 em pastas separadas usando o mesmo codec.pt, dados e seeds; o piloto de tons não demonstrou qualidade de voz nem eliminou overfitting.
