# TalkWeave — Da onda à conversa

Tutorial em português. Edição 01 — 8 de novembro de 2026.

Código original MIT; fontes externas conservam seus termos.


## 01 · O projeto que vamos construir

Uma rede pequena, compreensível e executável. Depois, um caminho para voz de alta qualidade.

O objetivo é construir uma rede causal que escuta áudio em blocos e produz áudio, texto e probabilidades de eventos a cada passo. Ela não precisa esperar o usuário terminar para continuar computando. Você poderá inspecionar o modelo, treinar todos os pesos do laboratório e ouvir a saída. A implementação começa sem pesos pré-treinados e não exige API paga.

A progressão segue a prática de livros como Python Machine Learning: representar dados, criar uma referência simples, separar treino e teste, otimizar uma função de perda e analisar erros antes de aumentar o modelo. Os exemplos são originais; o manual não reproduz o livro. Python básico, operações com arrays e noções de derivadas ajudam; cada tensor importante tem sua forma explicada.

Áudio usuário → codec → embeddings → GRU causal. História do agente → embeddings → GRU. Saídas: áudio do agente, texto usuário, texto agente, eventos. A saída anterior realimenta a rede.


### Três resultados diferentes

| Trilha | Você constrói | Limite real |
| --- | --- | --- |
| 01 · CPU / do zero | Logística NumPy, codec VQ, GRU causal, áudio/texto/eventos e streaming local. | Modelo didático. Precisa de dados humanos para aprender voz. |
| 02 · Projeto restrito | Um conjunto pequeno de comandos e respostas, treinado com suas gravações. | Generalização limitada ao vocabulário, pessoas e ambientes avaliados. |
| 03 · Voz avançada | Adaptação de uma base como PersonaPlex, com avaliação PT/EN. | Pesos maiores e GPU; português exige validação e não vem garantido. |

> Full-duplex tem duas camadas: transporte captura/reproduz ao mesmo tempo; o modelo também deve consumir a entrada enquanto gera a própria fala. Um sistema que só dispara depois do silêncio não demonstra essa segunda propriedade.


### Contrato de conclusão

Você termina a primeira trilha quando executa os testes, treina codec e modelo, gera um WAV com realimentação das próprias previsões e mede custo por quadro. Para concluir um produto de voz, acrescente um teste humano fora do treino: transcrição, risadas e ruídos, interrupções, inteligibilidade e tarefa concluída. Um script que roda não comprova que o modelo entende português.

```bash
python src/baseline.py
python src/fixture.py
python -m pytest -q
```

O que observar: Primeiro marco: ambiente funciona, dados têm formato correto e os testes de causalidade passam.

> Os exemplos de áudio incluídos no pacote são tons sintéticos. Não existe demonstração escondida de voz natural ou conversa treinada. Os laboratórios visuais são simulações explicitamente identificadas.


Antes de seguir:
- [ ] Entendi a diferença entre execução e qualidade.
- [ ] Baixei o projeto e conheço o resultado esperado.

Fonte: [Python Machine Learning · código oficial](https://github.com/rasbt/python-machine-learning-book-3rd-edition)

Fonte: [Moshi · artigo de arquitetura](https://arxiv.org/abs/2410.00037)


## 02 · Prepare um computador comum

Comece por CPU. Adicione uma GPU só quando a medição justificar.

Use Linux, macOS ou Windows com Python 3.10 ou superior. Como ponto de partida, um notebook com quatro núcleos e 8 GB de RAM costuma ser adequado para os exemplos pequenos; isso é uma orientação de capacidade, não uma configuração testada em todos os computadores. Um headset ajuda mais que uma GPU a evitar eco no primeiro experimento. Reserve alguns gigabytes para Python, PyTorch e os áudios.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

O que observar: Windows PowerShell: .venv\Scripts\Activate.ps1. Execute os comandos na pasta extraída do ZIP.

Não instale bibliotecas no ambiente que atende outros projetos. Os intervalos de versões no requirements.txt evitam trocas maiores, mas não são um lock universal. Após a instalação bem-sucedida, rode pip freeze e guarde o resultado junto ao experimento. O pacote fornece um arquivo com as versões efetivamente usadas na verificação deste tutorial.

```bash
python -m pip freeze > environment-lock.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

O que observar: False para CUDA é normal na trilha CPU. Nenhum exercício inicial depende de CUDA.


### Instalação do microfone

```bash
python -m pip install sounddevice
python -m sounddevice
python src/infer.py --microphone
```

Arquivo completo: src/infer.py

O que observar: A última linha exige codec.pt e duplex.pt treinados. Use Ctrl+C para encerrar.

Em Linux, a biblioteca PortAudio pode exigir libportaudio2 pelo gerenciador de pacotes. Se nenhum dispositivo aparece, resolva permissões e seleção de entrada antes de mexer na rede neural. O script local não oferece cancelamento de eco; comece com fones e volume baixo.

| Sintoma | Verifique primeiro |
| --- | --- |
| ModuleNotFoundError | O Python executado pertence à .venv? |
| No default input device | A entrada existe, está autorizada e é o dispositivo padrão? |
| Checkpoint não encontrado | Os dois estágios do treino foram executados no mesmo --output? |
| Treino lento em CPU | Use --threads 2, segmentos curtos e confira disputa com outros processos. |


Antes de seguir:
- [ ] Criei um ambiente separado.
- [ ] Importei NumPy, SoundFile e PyTorch.
- [ ] Os testes passam antes de treinar.

Fonte: [PyTorch · treinamento e otimização](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)


## 03 · Do ar aos números

Amostras, relógio, energia, espectro e os primeiros erros de áudio.

Um microfone transforma pressão em um sinal elétrico; o conversor produz amostras. A 16.000 Hz, um segundo contém 16.000 números por canal. O laboratório trabalha em mono, float32 na faixa de −1 a 1. Um bloco de 320 amostras dura 20 ms: 320 ÷ 16.000. Receber um bloco completo já implica aguardar esse intervalo; custo de inferência não é a latência total percebida.

```python
import numpy as np
x = np.zeros(320, dtype=np.float32)
rms = np.sqrt(np.mean(x * x) + 1e-12)
dbfs = 20 * np.log10(rms)
zcr = np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))
spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
```

O que observar: RMS mede energia, ZCR mede mudanças de sinal e FFT descreve frequências. Nenhum deles sozinho identifica palavras ou risadas.

Experimento interativo no site: senoide+ruído, RMS/ZCR calculados. O script baseline.py contém atributos equivalentes. Não identifica fala.


### Não confunda tamanho de bloco com modelo

O quadro de 20 ms é uma escolha deste laboratório, não uma exigência de qualquer modelo de voz. Mimi/PersonaPlex usam áudio de 24 kHz e outro relógio de codec. Um adaptador precisa fazer reamostragem com estado, juntar blocos e preservar os tempos. Trocar apenas o número da taxa de amostragem muda a velocidade da voz e corrompe o significado temporal.


### Faça estas três inspeções antes do treino

Ouça um arquivo inteiro para detectar cortes e canais trocados. Conte amostras e verifique se o relógio coincide com as anotações. Meça clipping: muitas amostras próximas de ±1 indicam saturação, que não se corrige diminuindo o volume depois. Remover o componente DC pode ajudar; normalização independente de cada quadro destrói a dinâmica e torna silêncio parecido com fala.

```python
import soundfile as sf
x, sr = sf.read("gravacao.wav", always_2d=True)
print({"segundos": len(x)/sr, "canais": x.shape[1],
       "clipping": float((np.abs(x) >= .999).mean())})
```

O que observar: A duração deve coincidir com a anotação. O clipping precisa ser investigado na captura, não escondido por ganho posterior.

> O botão de áudio do laboratório gera uma senoide local. A curva de energia é cálculo real, mas os rótulos não são inferência de fala.


Antes de seguir:
- [ ] Calculei a duração de um quadro.
- [ ] Conferi taxa, canais e clipping.
- [ ] Sei por que energia não é reconhecimento de fala.

Fonte: [AudioWorklet · MDN](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet)


## 04 · Monte dados que ensinem a conversar

Dois canais sincronizados, palavras alinhadas e eventos que não viram texto.

A qualidade começa nos exemplos. Para um piloto de tarefa restrita, escolha cinco a dez intenções e poucas respostas, grave variações, pessoas diferentes e silêncio real. Esse recorte torna a investigação barata; não existe um número mágico de minutos que garanta qualidade. Aumente os dados quando os erros em pessoas novas mostrarem necessidade. Áudio de leitura ajuda a aprender voz, mas não ensina interrupção e escuta simultânea.

Cada exemplo tem dois WAVs mono sincronizados: usuário e agente. Preserve o silêncio de ambos e a sobreposição. Nunca concatene os turnos removendo pausas. Se grave apenas uma mistura de alto-falantes, você já introduziu um problema extra de separação/diarização que este pacote não resolve. Prefira uma interface com dois canais ou uma gravação remota que exporte trilhas sincronizadas.

```bash
python src/record.py --input conversa-stereo.wav \
  --session conversa001 --speakers ana bia --split train
```

Arquivo completo: src/record.py

O que observar: Cria data/conversa001-user.wav e data/conversa001-agent.wav e adiciona uma linha ao manifesto. Entrada 1=usuário; entrada 2=agente.

```json
{"session":"001","speakers":["ana","bia"],"language":"pt",
 "user":"001-user.wav","agent":"001-agent.wav",
 "user_words":[{"start":0.2,"end":0.6,"text":"olá "}],
 "agent_words":[{"start":0.9,"end":1.4,"text":"oi "}],
 "events":[{"start":0.2,"end":0.6,"label":"speech"},
           {"start":1.6,"end":1.9,"label":"laughter"}]}
```

O que observar: No arquivo JSONL, cada objeto precisa ocupar uma única linha. Caminhos são relativos à pasta do manifesto.


### Como anotar sem infraestrutura cara

Abra as trilhas no Audacity, escute devagar e marque início e fim das palavras e eventos. Exporte os tempos e preencha o JSON. Use espaços no final das palavras para reconstruir frases. O alinhador do pacote distribui caracteres pelo intervalo já anotado; ele não descobre palavras nem tempos. Riso recebe laughter e não uma palavra inventada. Fala com ruído pode receber speech e noise ao mesmo tempo.

Nesta implementação, cada sessão de treino tem até seis segundos e até um caractere por quadro. Divida gravações longas preservando contexto e rebaseando os timestamps; se a fala exceder a resolução, adote subwords em uma extensão. O carregador recusa colisões e caracteres desconhecidos, em vez de treinar com rótulos silenciosamente errados.

| Conjunto | Separação correta | Uso |
| --- | --- | --- |
| Treino | Pessoas e sessões do treino | Ajustar pesos e estatísticas. |
| Validação | Pessoas não usadas no treino | Escolher arquitetura, limiar e checkpoint. |
| Teste | Pessoas e sessões ainda intocadas | Relatar uma medição final com incerteza. |

> Dados pessoais e vozes precisam de autorização de uso compatível com o projeto. Os arquivos ficam locais neste laboratório. CORAA e LibriTTS são referências de fala; não substituem um corpus próprio de diálogo com sobreposição. Confira os termos de cada conjunto.


Antes de seguir:
- [ ] Anotei palavras e eventos em uma sessão real.
- [ ] Preservei origem temporal e silêncios.
- [ ] Separei as duas pessoas de cada sessão entre os conjuntos.

Fonte: [CORAA · NILC](https://github.com/nilc-nlp/CORAA)

Fonte: [LibriTTS · OpenSLR](https://www.openslr.org/60/)


## 05 · Seu primeiro modelo, sem caixa-preta

Regressão logística, gradiente, normalização e regularização em NumPy.

Antes de uma rede recorrente, construa um classificador que você consegue depurar. Usaremos dois atributos: energia RMS e taxa de cruzamentos por zero. A regressão logística calcula z = wᵀx + b e p = 1/(1 + exp(−z)). O treinamento diminui a entropia cruzada; o gradiente em relação a z é p − y. A simplicidade torna visíveis erros de representação e vazamento.

```python
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
Z = (X_train - mean) / std
p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
error = p - y_train
w -= lr * (Z.T @ error / len(Z) + l2 * w)
b -= lr * error.mean()
```

Arquivo completo: src/baseline.py

O que observar: A média e o desvio vêm SOMENTE do treino. Guarde-os e reutilize na validação e inferência.

Experimento interativo no site: descida de gradiente real sobre dados sintéticos. Execute baseline.py para a versão NumPy.


### Execute o exercício no seu computador

```bash
python src/baseline.py
```

Arquivo completo: src/baseline.py

O que observar: O teste sintético incluído separa tons e ruído. Acurácia alta nesse exercício não indica detecção confiável de voz humana.

O laboratório interativo também usa dados sintéticos: uma distribuição de tons e outra de ruídos, com separação entre treino e teste. Ajuste a taxa de aprendizado para ver o efeito na curva. A implementação calcula os gradientes de fato no navegador. A linha de decisão é uma hipótese aprendida, não uma regra universal sobre sons.


### Como transformar em experimento útil

Substitua as amostras artificiais por janelas anotadas de gravações reais. Compare um limiar de energia, logística e uma pequena rede com as mesmas partições. Inspecione falsos positivos: ventilador, teclado, música e risadas. Se RMS/ZCR não separam as classes, treinar mais não cria a informação ausente; use log-mel/espectro ou uma CNN pequena. Salve scaler, pesos e versão dos dados.

> O método é inspirado nos fundamentos de aprendizado supervisionado usados em Python Machine Learning. Ele não transforma uma regressão logística em um reconhecedor de conversação.


Antes de seguir:
- [ ] Entendi o gradiente p − y.
- [ ] Padronizei só com dados do treino.
- [ ] Comparei com uma referência simples.

Fonte: [Python Machine Learning · código oficial](https://github.com/rasbt/python-machine-learning-book-3rd-edition)


## 06 · Fala, risada, ruído e sobreposição

Classifique eventos em paralelo. Não force todo som a virar uma palavra.

VAD responde se há atividade de voz; ASR tenta reconhecer conteúdo linguístico; detecção de eventos responde quais sons estão presentes. Essas perguntas são diferentes. Uma risada pode disparar um VAD e não conter uma palavra. Fala e ruído podem coexistir. Por isso a cabeça de eventos usa cinco saídas sigmoid independentes: speech, laughter, noise, cough e music.

```python
event_logits = model.event_head(hidden)
probabilities = event_logits.sigmoid()
loss = torch.nn.functional.binary_cross_entropy_with_logits(
    event_logits, multi_hot_labels
)
```

Arquivo completo: src/core.py

O que observar: Exemplo apenas ilustrativo: speech=0.91 e noise=0.78 podem ocorrer juntos. Esses números não foram medidos em áudio real.


### Defina o que cada rótulo significa

Use uma política de anotação: respiração não vira speech automaticamente; laughter deve cobrir o episódio de riso; noise significa interferência ambiental definida no seu conjunto. Um trecho com fala e riso pode ter os dois rótulos. Os eventos no manifesto se referem ao canal do usuário. Qualidade limpa/suja é outro atributo: não confunda ausência de noise com prova de áudio limpo.

O pacote começa com limiar 0,5 para demonstrar o mecanismo, não por ser ótimo. Selecione limiares por classe na validação usando precision/recall. Como risadas podem ser raras, acurácia agregada engana: prever sempre “sem risada” pode parecer ótimo. Relate F1 por classe, distribuição e falsos alarmes por minuto. Se uma classe não tem exemplos positivos no teste, não há evidência para avaliá-la.

| Erro | Consequência | Como examinar |
| --- | --- | --- |
| Risada → palavra | Texto alucinado e ferramenta acionada por engano. | Separe áudio de eventos e intenção textual. |
| Ruído → speech | Interrupções falsas. | Teste teclado, trânsito e eco do agente. |
| Riso misturado à fala → silêncio | Perda de conteúdo. | Anote sobreposição e preserve a trilha original. |

Silero e PANNs são referências externas para comparar tarefas, não componentes obrigatórios desta rede. A ontologia do AudioSet ajuda a escolher nomes consistentes. Para expandir de cinco para mais classes, altere EVENTS, anote dados, ajuste as dimensões e retreine; não apenas mude a legenda na interface.


Antes de seguir:
- [ ] Defini uma política de rótulos.
- [ ] Incluí exemplos positivos e negativos de cada classe.
- [ ] Medi falsos alarmes além de acurácia.

Fonte: [Silero VAD · repositório oficial](https://github.com/snakers4/silero-vad)

Fonte: [AudioSet · ontologia de sons](https://research.google.com/audioset/ontology/index.html)

Fonte: [PANNs · código dos autores](https://github.com/qiuqiangkong/audioset_tagging_cnn)


## 07 · Limpeza sem apagar informação

Eco, ruído e fala sobreposta pedem tratamentos diferentes.

Mantenha a gravação original. Crie uma derivação para reconhecimento e outra para preservar eventos, se necessário. Um denoiser agressivo pode remover risadas fracas e transformar consoantes; se você treinou com uma versão e executa com outra, criou mudança de distribuição. Compare sempre o mesmo áudio antes e depois e meça o efeito sobre reconhecimento e eventos.


### Cancelamento de eco não é supressão de ruído

Eco é a própria voz do agente saindo do alto-falante e voltando ao microfone. Um AEC usa uma referência do sinal reproduzido e estima sua transformação acústica. Ruído pode ser ventilação ou tráfego, sem referência exata. Fala de outra pessoa é uma fonte estruturada, não simplesmente um ruído estacionário. O runner local deste projeto usa fones e não contém AEC.

```javascript
const stream = await navigator.mediaDevices.getUserMedia({
  audio: {echoCancellation: true, noiseSuppression: true,
          autoGainControl: false, channelCount: 1}
});
console.log(stream.getAudioTracks()[0].getSettings());
```

O que observar: Exemplo para uma extensão no navegador. São restrições solicitadas: confira o que o navegador realmente aplicou. Não é um AEC próprio implementado.

O código JavaScript acima é um ponto de integração, não um frontend remoto pronto. Em produção, capture com AudioWorklet, mantenha referência de reprodução e trate reamostragem e relógios. Para o laboratório barato, fones eliminam grande parte do retorno acústico e reduzem variáveis durante os testes.


### Protocolo de comparação

Grave fala limpa, fala+ruído e fala+reprodução do agente, em volumes representativos. Compare o texto e a detecção de risadas antes/depois de cada transformação. Se houver gravação limpa de referência, use medidas de reconstrução; sem referência, evite inventar SNR verdadeiro. Para mistura controlada, ajuste o ganho do ruído pela relação entre energias e pelo SNR desejado.

```python
signal_power = np.mean(clean ** 2) + 1e-12
noise_power = np.mean(noise ** 2) + 1e-12
gain = np.sqrt(signal_power / (noise_power * 10**(snr_db/10)))
mixed = clean + gain * noise
```

O que observar: Evite clipping. Mistura controlada serve para medir robustez; não substitui ruído e reverberação reais.

> RNNoise é uma referência de supressão neural de ruído. Usá-lo exige respeitar sua interface e taxa de áudio; ele não resolve sozinho o eco do agente nem a separação de falantes.


Antes de seguir:
- [ ] Preservei os originais.
- [ ] Usei fones no primeiro teste.
- [ ] Comparei eventos e ASR antes/depois de limpar.

Fonte: [RNNoise · Xiph](https://github.com/xiph/rnnoise)

Fonte: [AudioWorklet · MDN](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet)


## 08 · Um codec neural feito por você

Encoder → quantização vetorial → decoder. Treine antes de gerar respostas.

Uma rede de diálogo não precisa prever cada amostra de 16 kHz diretamente. O codec comprime um quadro em um índice de uma tabela aprendida. Nosso encoder recebe [B,320], projeta para [B,32] e escolhe o vetor mais próximo entre 256 embeddings. O decoder transforma o embedding em 320 amostras. O orçamento é 8 bits × 50 quadros = 400 bits/s, sem cabeçalhos: muito restritivo para voz fiel.

```python
z = self.enc(x)
ids = self.nearest(z)
q = self.book(ids)
straight = z + (q - z).detach()
reconstruction = self.dec(straight)
vq_loss = F.mse_loss(q, z.detach()) + .25 * F.mse_loss(z, q.detach())
```

Arquivo completo: src/core.py

O que observar: A passagem forward usa o vetor quantizado. O straight-through aproxima o gradiente para o encoder; a tabela aprende pela perda VQ.

A perda combina erro de reconstrução, erro simples no espectro e regularização VQ. Não há promessa de fidelidade perceptual. O codec de um único codebook pode colapsar, usar poucos índices e produzir cliques nas fronteiras dos quadros. Esses são limites importantes do exercício, não problemas resolvidos por uma interface bonita.

```bash
python src/train.py codec --manifest data/train.jsonl \
  --validation data/validation.jsonl --steps 100
python src/reconstruct.py --input data/validation-user.wav \
  --output reconstructed.wav
```

Arquivo completo: src/reconstruct.py

O que observar: Ouça original e reconstrução. Se o codec não preserva palavras e risadas, o gerador posterior não pode recuperá-las de forma confiável.


### O que observar

Confira quantos códigos foram usados, compare perdas de treino/validação e ouça pessoas fora do treino. Um decoder que só produz silêncio pode ter erro médio baixo se os dados têm muito silêncio. Balanceie as janelas e inspecione períodos ativos. Não avalie qualidade só por MSE: um deslocamento de fase pequeno pode dar erro alto e ainda soar parecido; ruído médio pode dar erro modesto e soar ruim.


### Como subir a qualidade

Evolua para convoluções causais com estado, quantização residual com vários codebooks e perdas STFT multiescala, ou substitua pelo codec pré-treinado Mimi na trilha avançada. Essa substituição muda taxa, vocabulário e quantidade de códigos por quadro. Todos os alvos precisam ser recodificados e a rede de diálogo adaptada. Preserve um teste de causalidade para evitar que convoluções leiam o futuro.

> O codec aqui opera sobre o quadro atual completo. Assim, existe pelo menos a espera do quadro. A rede não usa quadros futuros; isso não significa latência zero.


Antes de seguir:
- [ ] Treinei o codec antes do diálogo.
- [ ] Ouvi a reconstrução.
- [ ] Inspecionei códigos usados e silêncio.

Fonte: [Moshi · artigo de arquitetura](https://arxiv.org/abs/2410.00037)


## 09 · Reconheça texto sem inventar palavras

Alinhamento supervisionado, blanks e avaliação por erros de palavras.

A cabeça de texto do usuário recebe o estado causal da GRU e prevê um caractere por quadro ou um blank. A cabeça do agente prevê seu próprio texto. São dois fluxos distintos: o texto reconhecido do usuário não deve ser confundido com a resposta. Um único fluxo de texto não representa duas pessoas falando ao mesmo tempo.

```python
user_logits = self.user_text(hidden)
agent_logits = self.agent_text(hidden)
weights = torch.ones(len(CHARS))
weights[0] = 0.05  # blank frequente; valor inicial a validar
loss = F.cross_entropy(user_logits, target_chars, weight=weights)
```

Arquivo completo: src/core.py

O que observar: Nos tensores do modelo, logits têm forma [batch, tempo, vocabulário]. A função de treino recorta um exemplo para usar [tempo, vocabulário].


### Por que não usamos CTC neste primeiro código

O laboratório já exige tempos anotados e usa entropia cruzada por quadro. Blank significa “nenhum caractere neste instante”. Ao decodificar, retiramos blanks e preservamos repetições: “rr” contém duas letras. CTC seria uma alternativa quando há transcrições sem alinhamento fino, mas sua regra de colapsar repetições é diferente; não a aplique a esta saída. Migrar para CTC exige outra loss e outro decoder.

```python
def token_text(tokens):
    return "".join(CHARS[int(i)] for i in tokens)

# O índice zero mapeia para string vazia.
# Não fazemos deduplicação: isso apagaria letras legítimas.
```

Arquivo completo: src/core.py

O que observar: Se o modelo repete caracteres demais, corrija treinamento/alinhamento. Não esconda o erro com deduplicação indiscriminada.

O vocabulário inclui caracteres frequentes de português e inglês, com acentos. Normalizamos em Unicode NFC e minúsculas. Um caractere desconhecido produz erro explícito. Nomes, abreviações e números devem aparecer no conjunto de avaliação. Não traduza o texto na avaliação se o objetivo é medir reconhecimento fiel.

Experimento interativo no site: distância de edição entre textos. core.wer fornece versão Python.


### Comparação externa opcional

Um reconhecedor pronto, como faster-whisper, pode transcrever os mesmos áudios offline para fornecer uma referência prática. Essa comparação não transforma a arquitetura em uma cascata nem substitui a cabeça treinada. Whisper com janelas e revisão de hipóteses também não é automaticamente um reconhecedor causal com latência de 20 ms. Compare fidelidade e latência separadamente.


Antes de seguir:
- [ ] Distingui texto do usuário e do agente.
- [ ] Preservei letras repetidas.
- [ ] Calculei WER fora do treino.

Fonte: [faster-whisper · repositório oficial](https://github.com/SYSTRAN/faster-whisper)


## 10 · Construa a rede full-duplex

Uma GRU causal, estado persistente e quatro saídas por instante.

No passo t, o modelo recebe o código do áudio atual do usuário u[t], o código anterior de sua própria fala a[t−1] e o caractere anterior do agente. A GRU atualiza h[t]. Deste estado saem texto reconhecido do usuário, texto do agente e eventos. A cabeça acústica usa h[t] e o texto atual do agente para prever a[t]. Não existe regra de esperar fim de frase para avançar.

Áudio usuário → codec → embeddings → GRU causal. História do agente → embeddings → GRU. Saídas: áudio do agente, texto usuário, texto agente, eventos. A saída anterior realimenta a rede.

```python
inputs = torch.cat([audio_embedding(user_code),
                    audio_embedding(previous_agent_code),
                    text_embedding(previous_agent_text)], dim=-1)
h, state = rnn(inputs, state)
user_text = user_head(h)
agent_text = agent_head(h)
events = event_head(h)
audio = audio_head(torch.cat([h, text_embedding(agent_token)], -1))
```

Arquivo completo: src/core.py

O que observar: Entrada da GRU: [B,T,120]. Estado padrão: [1,B,128]. Cada cabeça compartilha história, mas resolve uma tarefa diferente.


### Por que GRU em vez de um transformer grande

A GRU mantém um vetor de estado compacto e executa um passo com custo previsível. É adequada para aprender causalidade e rodar experimentos em CPU. Não oferece a capacidade contextual de um modelo de bilhões de parâmetros. Escolher GRU reduz custo; não cria conhecimento geral nem conversação fluente por si só.

A rede vê o usuário inclusive quando está emitindo áudio. Aprender a parar, continuar ou dar uma resposta curta depende das gravações e da perda, não apenas da conectividade. Inclua exemplos de interrupção legítima, ruído que deve ser ignorado e concordância curta que não deve tomar o turno.


### Prove a causalidade

```python
before = model(user, previous_audio, previous_text)[0]
changed = user.clone()
changed[:, 10:] = 255
after = model(changed, previous_audio, previous_text)[0]
torch.testing.assert_close(before[:, :10], after[:, :10])
```

Arquivo completo: tests/test_core.py

O que observar: Mudar o futuro não pode alterar saídas anteriores. Outro teste compara execução da sequência inteira com execução quadro a quadro usando o mesmo estado.

> Transporte simultâneo e estrutura causal estão implementados. A adequação conversacional só será aprendida com dados e demonstrada em testes humanos. A rede pequena não é uma reprodução de Moshi ou PersonaPlex.


Antes de seguir:
- [ ] Identifiquei cada entrada e cada saída.
- [ ] Preservei o estado entre quadros.
- [ ] Executei os testes de causalidade e equivalência.

Fonte: [Moshi · artigo de arquitetura](https://arxiv.org/abs/2410.00037)


## 11 · Treine e reconheça overfitting

Loss composta, gradientes, validação e geração livre.

O treino tem dois estágios. Primeiro o codec aprende a reconstruir áudio. Depois congelamos o codec e treinamos a GRU para prever códigos, texto e eventos. Se você muda a tabela de códigos durante o segundo estágio, muda também o significado dos alvos. Congele os pesos e guarde a versão usada para codificar os dados.

```python
loss = (audio_ce
        + 0.5 * user_text_ce
        + 0.5 * agent_text_ce
        + 0.2 * event_bce)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

Arquivo completo: src/train.py

O que observar: Pesos são valores iniciais didáticos. Ajuste na validação; não compare numericamente losses de tarefas diferentes sem considerar escala e frequência.

O otimizador é AdamW, com learning rate inicial 0,001 no modelo pequeno. Clipping limita a norma do gradiente. O texto usa menor peso para blank e o áudio dá menor peso ao código de silêncio. Isso reduz o incentivo a responder sempre com nada, mas não resolve desbalanceamento sozinho. Um erro não finito interrompe o treino para evitar checkpoints corrompidos.

```bash
python src/train.py duplex --manifest data/train.jsonl \
  --validation data/validation.jsonl --steps 300 --threads 2
python src/infer.py --input data/validation-user.wav --output reply.wav
python src/evaluate.py --manifest data/validation.jsonl --report reply.wav.json
```

Arquivo completo: src/train.py

O que observar: Treino salva runs/duplex.pt e métricas JSON. A avaliação usa reply.wav.json produzido por geração livre, sem fornecer a resposta correta ao modelo.


### Dropout: regularizar sem desligar o streaming

Dropout zera aleatoriamente uma fração das ativações durante o treino e reescala as restantes. Não remove neurônios permanentemente. Isso pode reduzir coadaptação e overfitting; não impede decorar sozinho e não substitui dados variados. O modelo agora aceita --dropout no estágio duplex, mantendo 0 como padrão para preservar a referência publicada.

```python
model = Duplex(dropout=0.1)
model.train()  # activation masks are sampled during training
# After training:
model.eval()   # dropout is disabled for inference
```

Arquivo completo: src/core.py

O que observar: A implementação aplica dropout aos embeddings de entrada e à saída da GRU. Ela não zera diretamente o estado recorrente nem corta o microfone.

A GRU do laboratório tem uma camada. O parâmetro dropout do próprio nn.GRU atua entre camadas e não é uma solução para essa configuração; por isso usamos módulos nn.Dropout explícitos. O código aplica máscaras independentes às ativações, sem afirmar que implementa variational dropout ou dropout recorrente.

```bash
python src/train.py duplex --manifest data/train.jsonl --validation data/validation.jsonl --steps 100 --dropout 0.1
```

Arquivo completo: src/train.py

O que observar: Comece comparando 0, 0.1 e, se necessário, 0.2. Use pastas separadas com o mesmo codec.pt para não sobrescrever os modelos que deseja comparar. O treinador salva duplex-config.json com o valor usado.

| Piloto sintético · 100 passos · seed 7 | Dropout 0 | Dropout 0.1 |
| --- | --- | --- |
| Loss de treino final | 0.420 | 0.541 |
| Loss de validação final | 4.783 | 4.698 |

Nesta comparação de tons, a validação ficou um pouco menor com 0.1, mas continuou piorando ao longo do treino. Uma seed e duas sessões artificiais não comprovam ganho de qualidade de voz nem resolvem o overfitting. Compare várias seeds, pessoas fora do treino e geração livre; uma loss de treino maior com dropout pode ser esperada.

> model.eval() desliga dropout; torch.no_grad() ou inference_mode() sozinhos não mudam o modo de treino. Na inferência o áudio deve continuar quadro a quadro, sem máscaras aleatórias. Taxas altas podem causar underfitting e prejudicar a representação acústica.

[{"step": 1, "loss": 9.4258451461792, "validation_loss": 9.170353889465332}, {"step": 25, "loss": 2.1212942600250244, "validation_loss": 3.037565231323242}, {"step": 50, "loss": 0.949921727180481, "validation_loss": 3.9162685871124268}, {"step": 75, "loss": 0.5910598039627075, "validation_loss": 4.634428024291992}, {"step": 100, "loss": 0.41977450251579285, "validation_loss": 4.782999515533447}]


### O exemplo real de verificação deste tutorial

Rodamos 100 passos de cada estágio em CPU com o fixture de tons. A loss de treino do duplex caiu de cerca de 9,43 para 0,42, enquanto a validação subiu para cerca de 4,78. A geração livre não produziu as palavras artificiais esperadas. Isso é um exemplo concreto de por que loss de treino baixa não é sucesso. As curvas e o áudio estão na área de resultados, sem seleção de um caso favorável.


### Teacher forcing e o erro que ele esconde

Durante o treino fornecemos áudio/texto anteriores verdadeiros. Durante a execução fornecemos as previsões anteriores da rede. Essa diferença se chama exposure bias e permite que erros se acumulem. Para uma extensão séria, compare scheduled sampling ou treinamento com perturbação dos históricos, sempre com geração livre na validação. Não use a sequência correta do agente para reportar desempenho de conversa.

O treinador carrega o conjunto na RAM e usa até os primeiros 300 quadros de cada exemplo. Por isso recorte sessões em exemplos de até seis segundos. Para dados longos, implemente shards, batches mascarados e treino em chunks com estado e truncamento de gradientes. O código inicial não é uma infraestrutura para centenas de horas.


Antes de seguir:
- [ ] Treinei os dois estágios.
- [ ] Comparei treino e validação.
- [ ] Escutei geração livre e identifiquei overfitting.

Fonte: [PyTorch · treinamento e otimização](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)


## 12 · Gere áudio e texto de resposta

Feche o ciclo: saída prevista volta como entrada no próximo passo.

Na inferência, codifique o quadro atual do usuário, atualize a GRU, escolha o texto do agente e o próximo código de áudio, decodifique e reproduza. Guarde estado, código e texto para o próximo tick. O pacote usa argmax para facilitar a reprodução dos testes. Sampling pode diversificar voz e conteúdo, mas também elevar ruído e instabilidade; avalie depois da referência determinística.

```python
audio, user_char, agent_char, probabilities = session.step(frame)
# session guarda hidden, previous_audio e previous_text
# frame: 320 amostras float32; audio: 320 amostras float32
```

Arquivo completo: src/infer.py

O que observar: O laço continua recebendo entrada enquanto produz saída. A cada passo a geração depende da própria história prevista.

```bash
python src/infer.py --input data/validation-user.wav --output reply.wav
python src/reconstruct.py --input data/validation-user.wav --output codec.wav
python src/infer.py --text "olá mundo" --seconds 3 --output text.wav
```

Arquivo completo: src/infer.py

O que observar: São três tarefas distintas: responder a áudio, reconstruir áudio e testar condicionamento textual. O terceiro modo é experimental; não é um TTS universal pronto.


### Texto para voz dentro da rede

A cabeça acústica é condicionada pelo caractere atual do agente. No treino, esse caractere é verdadeiro; na conversa, é previsto. O modo --text força uma sequência de caracteres com um relógio simples. Ele permite investigar a relação texto/áudio, mas não aprendeu duração, prosódia ou cobertura arbitrária de texto. Para um TTS completo, adicione treinamento com texto condicionado e modelagem de duração, ou uma arquitetura de tokens de texto/áudio apropriada.

Se a resposta fica muda, verifique ocupação de silêncio, frequências de blank e qualidade da reconstrução. Se sai ruído, teste codec isolado, compare ordens de canais e procure divergência entre treino e inferência. Se o texto é bom e áudio ruim, mais dados textuais não corrigem necessariamente o decoder acústico.


### Ouça os artefatos reais

Ouça downloads/validation-user.wav e downloads/reply.wav no site. Ambos contêm tons/saída sintética, sem fala humana.

> Os áudios desta verificação contêm tons e a saída do modelo treinado com eles. Eles comprovam execução do pipeline, não fala, risadas reconhecidas ou conversação em português. A avaliação de voz deve ser feita depois com dados humanos.


Antes de seguir:
- [ ] Gerei um WAV com inferência livre.
- [ ] Distingui resposta de reconstrução.
- [ ] Não tratei condicionamento textual experimental como TTS completo.


## 13 · Protocolo e áudio simultâneo

Relógio amostral, filas limitadas e reprodução contínua.

O runner de microfone usa uma callback do PortAudio para captura e reprodução simultâneas. A callback só copia entrada e retira saída de filas limitadas. Inferência, logs e disco ficam fora dela. Executar a rede dentro da callback cria travamentos quando um passo demora; a prioridade é nunca bloquear o relógio de áudio.

```python
incoming = queue.Queue(maxsize=8)
outgoing = queue.Queue(maxsize=8)
# Na callback: put_nowait / get_nowait.
# Fora dela: inferir e preencher a fila de reprodução.
# Se não houver saída, preencher com zeros e contar underrun.
```

Arquivo completo: src/infer.py

O que observar: O pacote conta overflow de entrada, underrun de saída e alertas do PortAudio. Fila acumulada aumenta latência mesmo quando a rede é rápida.


### Um envelope para transporte remoto

protocol.py implementa serialização binária, não um servidor WebSocket pronto. Cabeçalho little-endian: versão uint8, sequência uint64, relógio de amostras uint64 e quantidade uint16; depois PCM16 little-endian. A taxa e os canais precisam ser negociados por mensagem de início. O relógio é relativo à sessão, não horário de parede.

```python
from protocol import pack, unpack
packet = pack(sequence=7, sample_clock=960, samples=frame)
seq, clock, audio = unpack(packet)
```

Arquivo completo: src/protocol.py

O que observar: Cabeçalho: 19 bytes. Um quadro mono de 320 amostras PCM16 ocupa 640 bytes; total de 659 bytes antes do transporte.

Para uma extensão remota, negocie session_id, sample_rate, frame_samples e channels, valide tamanho máximo e rejeite formatos desconhecidos. WebSocket preserva ordem mas pode acumular atraso por retransmissão; sequência e relógio permitem diagnosticar lacunas e filas. Em rede adversa, WebRTC/Opus costuma ser uma alternativa a avaliar. Não misture mensagens de contexto com bytes PCM sem distinguir o tipo.

Simulação visual: captura permanece ativa enquanto agente e worker operam; interrupção invalida saída antiga. Não é inferência real.


### Interrupção não significa apagar toda a memória

Ao reconhecer uma interrupção válida, descarte áudio ainda não reproduzido do turno anterior e invalide resultados antigos por geração de turno. Continue recebendo o usuário. Um detector de energia sozinho não é suficiente: ruído e a voz do próprio agente podem disparar falso corte. No modelo neural, compare exemplos de interrupção e backchannel antes de adicionar uma regra externa.

> A linha do tempo acima é uma simulação de filas e concorrência, sem inferência de voz. O modelo executável do ZIP usa áudio local. Servidor remoto, jitter buffer adaptativo, reconexão e AEC são extensões documentadas, não entregues como prontas.


Antes de seguir:
- [ ] Mantive inferência fora da callback.
- [ ] Medi filas e underruns.
- [ ] Entendi o que o envelope implementa e o que falta ao transporte remoto.

Fonte: [AudioWorklet · MDN](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet)


## 14 · Meça precisão e qualidade de verdade

Um painel de métricas, gravações humanas e falhas reproduzíveis.

Precisão não é um único número. Um sistema pode transcrever bem e interromper a pessoa; pode gerar uma voz agradável com resposta errada. Defina uma matriz de condições antes de ajustar o modelo: português/inglês, sotaques, pessoas novas, ruído, risada durante fala, nomes, números e fala simultânea. Preserve um teste final sem uso em decisões.

| Propriedade | Medição | Armadilha |
| --- | --- | --- |
| Texto fiel | WER/CER, erros em nomes e números | Normalizar tudo pode esconder erro importante. |
| Eventos | F1 por classe, falsos alarmes/min, tolerância temporal | F1 por quadro não mede fronteiras de episódios. |
| Voz | Inteligibilidade e escuta humana cega | MSE de áudio não mede naturalidade. |
| Duplex | Tempo para ceder turno, falsos cortes, backchannels | Cortar todo áudio com energia parece rápido, mas é ruim. |
| Tempo real | RTF, p50/p95 por passo, filas e underruns | Um passo rápido não inclui captura, rede e playout. |
| Tarefa | Estado final verificado na aplicação | Dizer “feito” não comprova execução. |

```bash
python src/evaluate.py --manifest data/validation.jsonl --report reply.wav.json
```

Arquivo completo: src/evaluate.py

O que observar: O script fornece WER dos dois fluxos e F1 por quadro dos eventos com limiar 0,5. Não calcula naturalidade, fronteiras de eventos nem qualidade de interrupção.


### Como medir latência sem se enganar

RTF é tempo total de processamento dividido pela duração do áudio; menor que 1 indica capacidade média suficiente naquela execução. p95 do passo precisa caber no orçamento do quadro com margem, mas uma fila crescente ainda pode violar a experiência. Registre captura, chegada, início/fim da inferência, entrada na fila e playout com relógios monotônicos. Para ponta a ponta, use gravação de loopback e um estímulo identificável.

Metas iniciais de projeto podem ser, por exemplo, RTF abaixo de 1 e zero perda de quadros na sessão local. São critérios de engenharia, não resultados prometidos. Defina metas de WER e eventos a partir do risco da tarefa e da referência existente. Um comando de pagamento merece outro tratamento que um exercício de cores.


### Escuta humana barata e útil

Apresente amostras embaralhadas sem dizer qual modelo as gerou. Peça que a pessoa transcreva o que ouviu e avalie inteligibilidade, naturalidade e adequação. Registre tamanho da amostra e dispersão, não apenas média. Para comparar versões, mantenha prompts, áudios e condições iguais. Não use só avaliação automática por outro modelo, pois ela pode não perceber artefatos acústicos ou sotaques mal atendidos.

> A verificação CPU do pacote não incluiu microfone real, GPU ou vozes humanas. Os arquivos de métricas registram exatamente o experimento sintético. Essa limitação está preservada nos resultados.


Antes de seguir:
- [ ] Separei qualidade de texto, voz, eventos e turno.
- [ ] Medi inferência livre.
- [ ] Preservei teste por pessoas novas.


## 15 · Faça caber no equipamento barato

Reduza escopo e meça antes de comprar hardware.

O primeiro ganho vem de escolher uma tarefa pequena. Um classificador de sons e uma rede de respostas limitadas exigem menos dados e capacidade que uma conversa aberta. Não existe técnica de livro que transforme um modelo minúsculo, sem dados, em um agente geral com qualidade de fronteira. Você pode construir o mecanismo completo em CPU e escalar só a parte que comprovadamente limita o resultado.

| Equipamento | Caminho sugerido | Não presumir |
| --- | --- | --- |
| CPU / cerca de 8 GB RAM | NumPy, codec e GRU pequenos; sessões curtas, treino em etapas. | Conversa geral natural ou desempenho em todos os notebooks. |
| GPU modesta / 6–12 GB | Aumentar batches e redes pequenas após validar dados. | Que PersonaPlex 7B com treino completo cabe sem ajustes. |
| 3× 96 GB da sua máquina | Experimentos de adaptação de base e avaliação simultânea. | Que mais VRAM resolve falta de diálogo PT ou erros de anotação. |

Pesos FP32=4B/parâmetro; com gradientes+Adam ≈16B/parâmetro antes de ativações. WAV estéreo16k PCM16 = 230.4MB/h em unidade decimal.


### Otimizações na ordem certa

Primeiro: evite reprocessar áudio e recodifique datasets uma vez quando o codec está congelado. Segundo: diminua threads excessivas, cópias e logs no caminho crítico. Terceiro: use inferência sem autograd e batch só quando não prejudicar a latência de uma sessão. Só então avalie quantização, exportação e compilação, comparando saída e tempo real.

Quantização de pesos reduz memória, mas operações e plataformas diferentes podem ter suporte desigual. Quantização dinâmica de camadas lineares/GRU em CPU é uma investigação possível; não é ativada no pacote nem acompanhada de benchmark. Em voz, valide os áudios e eventos após cada alteração. Redução de memória sem voz inteligível não é ganho útil.


### O que o treino realmente guarda

Além de pesos há gradientes, estados do otimizador, ativações e os dados. Adam em precisão simples costuma exigir múltiplas cópias por parâmetro; a calculadora mostra só uma aproximação de estados, sem ativações. Nosso carregador usa RAM proporcional a todo o áudio, então mantenha um piloto pequeno e use shards quando crescer. O modelo pequeno cabe; um corpus inteiro pode não caber.

Comece sem gastar com APIs: o gerador de tons, regressão NumPy, treino e inferência são locais. O primeiro custo inevitável de uma aplicação de voz é obter e conferir bons dados. Uma interface de áudio melhor e anotações consistentes podem trazer mais benefício que aumentar o modelo prematuramente.


Antes de seguir:
- [ ] Defini uma tarefa restrita.
- [ ] Medi antes de otimizar.
- [ ] Incluí dados e ativações no planejamento.

Fonte: [PyTorch · treinamento e otimização](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)


## 16 · Quando subir para PersonaPlex

Adapte uma base forte sem confundir pesquisa com suporte comprovado em português.

O caminho avançado reaproveita um modelo com pesos abertos e competência conversacional prévia. PersonaPlex usa papel definido por texto e condicionamento de voz, em arquitetura full-duplex. Ele é muito maior que a GRU do laboratório e não é a trilha barata de construção do zero. As licenças de código, pesos e dados precisam ser tratadas separadamente.


### O que existe de concreto

Há um treinador comunitário em tin-computer/personaplex-finetune e um adapter publicado em inglês. Isso mostra uma rota de adaptação de domínio; não demonstra transferência linguística para PT-BR. Na discussão pública sobre português, um integrante da NVIDIA apontou a falta de dados de conversação. Portanto, um sprint de 48 horas deve ser tratado como experimento com critérios de aprovação, não como garantia de fluência.

```bash
git clone https://github.com/tin-computer/personaplex-finetune.git
cd personaplex-finetune
git rev-parse HEAD
# Siga docs/SETUP_GUIDE.md na revisão que você fixou.
# Use ambientes separados para preparo, treino e inferência.
```

O que observar: Esses comandos obtêm o código. Não substituem o setup específico de CUDA/torch/flash-attn nem executam treino automaticamente.

Antes de LoRA, teste o codec com áudio PT/EN e faça uma referência da base: reconhecimento, pronúncia, interrupção e latência. Prepare diálogo PT de dois canais e mantenha exemplos em inglês para reduzir esquecimento. Compare dois ajustes curtos em vez de gastar todas as GPUs em uma execução distribuída ainda não validada.


### Plano de 48 horas, condicionado aos dados

| Janela de trabalho | Saída verificável |
| --- | --- |
| 0–4 h | Ambiente compatível e base funcionando; versões fixadas. |
| 4–12 h | Amostras PT/EN anotadas e separadas por pessoa. |
| 12–32 h | Ajustes curtos e comparação de checkpoints em geração livre. |
| 32–48 h | Escuta cega, testes de interrupção e decisão de continuar ou mudar. |

Esse é um orçamento de trabalho, não tempo medido nas suas três Blackwell. PCIe com P2P permite comunicação, mas não equivale a NVLink. Memória disponível não garante kernels compatíveis: valide o suporte a sm_120 e preserve o ambiente vLLM existente. Não formate discos para seguir o tutorial; use uma pasta com espaço confirmado.

> O fork é experimental e relata dificuldades ao ajustar o depformer. A receita de domínio em inglês congela essa parte para preservar áudio; ela pode não ser suficiente para acrescentar uma língua. Não copie um número de rank ou passos como se fosse uma receita validada para português.


Antes de seguir:
- [ ] Conferi a revisão e a licença.
- [ ] Separei adaptação de domínio de nova língua.
- [ ] Tenho referência PT/EN antes do ajuste.

Fonte: [PersonaPlex · pesquisa NVIDIA](https://research.nvidia.com/labs/adlr/personaplex/)

Fonte: [PersonaPlex · fine-tuning comunitário](https://github.com/tin-computer/personaplex-finetune)

Fonte: [PersonaPlex · adaptação publicada em inglês](https://huggingface.co/demegire/personaplex-finetune-pharma)

Fonte: [PersonaPlex · discussão sobre português](https://huggingface.co/nvidia/personaplex-7b-v1/discussions/36)


## 17 · Conecte ações sem bloquear a voz

Executor assíncrono, confirmação de resultado e contexto com proveniência.

Uma rede de voz não precisa executar automação de navegador dentro do laço acústico. A tarefa pode rodar em um worker, enquanto a captura continua. O resultado deve trazer session_id, turn_id, tool_call_id, estado e evidência. Uma mensagem de texto reconhecida parcialmente não é autorização suficiente para qualquer ação; fragmentos podem mudar o sentido da frase.

```json
{"type":"tool.result","session_id":"s1","turn_id":12,
 "tool_call_id":"call-7","status":"verified",
 "data":{"input_value":""},
 "evidence":{"selector":"input[type=search]","checked":true}}
```

O que observar: Exemplo de contrato proposto. Este JSON não é um formato nativo do modelo pequeno nem uma ferramenta já conectada.


### Evite o “feito” sem execução

Ao limpar um campo de busca, acione a edição, leia o valor final e confirme a condição esperada. Um DOM sem nós visíveis não prova que o campo está vazio. Para abrir contato, compare o contato realmente aberto com o alvo. Registre falhas e não converta ausência de evidência em sucesso. Resultados que chegam após cancelamento precisam ser descartados ou claramente identificados como antigos.


### O que o PersonaPlex comunitário oferece

```json
{"type":"context","text":"Busca concluída: três contatos chamados Michael. Nenhum foi aberto ainda."}
```

O que observar: O fork documenta uma mensagem WebSocket de contexto. Isso é entrada de contexto, não tool calling nativo ou verificação de estado da aplicação.

Nesse fork, a injeção força silêncio do agente durante a leitura do contexto, preservando a entrada do usuário. Portanto não prometa fala contínua durante essa operação. Mantenha mensagens curtas e meça a pausa. O modelo pequeno do laboratório não possui esse canal: para adicioná-lo seria necessário definir condicionamento, preparar exemplos e treinar de modo correspondente.

O objetivo desta etapa é especificar a fronteira correta. Um agente de navegação completo, seletor de intenção e servidor remoto não acompanham o pacote de rede neural. O restante do tutorial é executável localmente; esta extensão precisa ser integrada à aplicação que você já usa. A voz só deve anunciar conclusão verificada pelo executor.


Antes de seguir:
- [ ] Separei ação de inferência acústica.
- [ ] Exigi evidência para sucesso.
- [ ] Identifiquei os componentes ainda não integrados.

Fonte: [PersonaPlex · fine-tuning comunitário](https://github.com/tin-computer/personaplex-finetune)


## 18 · Execute de ponta a ponta

Um roteiro único, arquivos completos e diagnóstico de falhas.


### Roteiro A · prove a engenharia em CPU

```bash
python src/baseline.py
python src/fixture.py
python -m pytest -q
python src/train.py codec --manifest data/train.jsonl --validation data/validation.jsonl --steps 100
python src/reconstruct.py --input data/validation-user.wav --output reconstructed.wav
python src/train.py duplex --manifest data/train.jsonl --validation data/validation.jsonl --steps 100
python src/infer.py --input data/validation-user.wav --output reply.wav
python src/evaluate.py --manifest data/validation.jsonl --report reply.wav.json
```

O que observar: Produz modelos, curvas JSON, reconstrução, resposta e relatório. Os dados deste roteiro são tons sintéticos.


### Roteiro B · comece o conjunto humano

Faça uma cópia separada da pasta data para não misturar fixtures e pessoas reais. Importe várias sessões estéreo de até seis segundos. Preencha palavras e eventos manualmente. Organize treino, validação e teste com pessoas diferentes. Treine codec, escute reconstrução e só avance se os detalhes que importam foram preservados. Treine o duplex e meça geração livre; aumente escopo apenas quando o conjunto atual generalizar.

```bash
python src/record.py --input sessao01.wav --session real001 --speakers ana bia --split train
# Edite data/train.jsonl para preencher palavras e eventos.
# Importe outras pessoas em validation e test; não reutilize o mesmo par.
python src/train.py codec --manifest data/train.jsonl --validation data/validation.jsonl --steps 1000
python src/reconstruct.py --input data/real001-user.wav
python src/train.py duplex --manifest data/train.jsonl --validation data/validation.jsonl --steps 1000
```

Arquivo completo: src/record.py

O que observar: 1000 passos é um ponto de experimento, não um requisito nem uma garantia. Inspecione curvas e geração antes de aumentar.

| Problema | Hipótese a testar |
| --- | --- |
| Validação piora enquanto treino melhora | Overfitting, poucas pessoas ou dados repetidos; reduza capacidade e melhore variedade. |
| Resposta livre diverge | Exposure bias; compare treino e geração sem históricos verdadeiros. |
| Tudo vira silêncio/blank | Desbalanceamento, loss ou alinhamento; examine contagens por classe. |
| Codec apaga fala | Capacidade acústica insuficiente; melhore codec antes do diálogo. |
| Modelo fala com ele mesmo | Eco/bleed de canais; use fones e confira trilhas. |
| Texto contém símbolos inesperados | Vocabulário/normalização inconsistente ou checkpoint errado. |
| Fila cresce | Processamento não acompanha relógio; meça p95 e callbacks. |


### Arquivos do projeto

Arquivos completos estão na pasta deste projeto e no visualizador do site.


### Registro de verificação

Executado em CPU · Python 3.12 · PyTorch 2.15.0+cpu
9 testes passaram: envelope PCM, alinhamento, causalidade,
equivalência streaming/sequência, gradiente do codec, WER/shift,
contrato Frame/Result, docstrings de uma linha e dropout treino/inferência.
Codec: 100 passos · Duplex: 100 passos · seed 7
Fixture: 2 sessões de tons, sem voz humana.
Geração livre: WAV e relatório JSON produzidos.
Texto do usuário/agente vazio no fixture de validação.
Não verificado: microfone físico, GPU, precisão de fala humana.
Tempo por quadro registrado é apenas inferência local deste experimento,
não inclui captura, rede ou reprodução e não é promessa de desempenho.

> “Completo” aqui significa que a implementação didática vem com todos os arquivos para instalar, preparar fixture, treinar, inferir e avaliar. Ela não entrega pesos de conversação geral nem comprovação de precisão em vozes humanas. As extensões avançadas são identificadas como tais.


Antes de seguir:
- [ ] Executei o roteiro sem pular etapas.
- [ ] Li os arquivos completos.
- [ ] Registrei limitações e próximos experimentos.


## 19 · Padrões de código e streaming

Pastas curtas, interfaces claras e uma interface de console fora do caminho crítico.

O projeto segue uma estrutura pequena e convencional: run.py na raiz; implementação em src/; configuração em cfg/; entradas em data/; pesos em runs/; registros em logs/; documentação em docs/; testes em tests/; experimentos descartáveis em work/. Nomes curtos devem continuar legíveis: Frame, Result, Engine, Codec e Session deixam as responsabilidades explícitas.

```text
run.py
src/     # implementação
cfg/     # configurações
data/    # entradas locais
runs/    # pesos e métricas
logs/    # evidências
docs/    # padrões e relatórios
tests/   # testes
work/    # experimentos isolados
```

O que observar: O runner cria sua própria pasta de experimento. Ele não substitui gravações humanas existentes nem modifica outro ambiente de inferência.

Console Rich: captura de uma saída real sobre tons sintéticos. Rode python run.py --ui ou python src/infer.py --microphone --ui para a interface local.


### Docstrings e interfaces

```python
class Engine(Protocol):
    """Define a causal frame processing boundary."""

    def process(self, frame: Frame) -> Result:
        """Consume one frame and return generated audio and measured outputs."""
        ...
```

Arquivo completo: src/api.py

O que observar: Funções e classes têm docstrings de uma linha em inglês. Os contratos públicos usam tipos; a UI depende do resultado, não dos detalhes da rede.

Use snake_case em funções, PascalCase em classes e constantes em UPPER_CASE. Separe captura, modelo, protocolo e apresentação. Uma abstração deve proteger uma fronteira útil; não crie uma hierarquia para uma única operação. O protocolo Engine permite trocar o modelo sem reescrever a captura ou inventar outro formato de resultado.


### Streaming puro, com orçamento explícito

Cada Frame carrega 320 amostras mono float32 a 16 kHz e uma sequência crescente. O modelo processa o quadro atual e o estado passado; não acumula uma frase antes de avançar. Result contém áudio gerado, textos, escores independentes de eventos e tempo de processamento medido. A callback só passa blocos por filas limitadas. Rede, disco, renderização e ferramentas permanecem fora dela.

```bash
python run.py --ui
# Depois de obter checkpoints adequados:
python src/infer.py --microphone --ui
```

Arquivo completo: src/ui.py

O que observar: O primeiro comando mostra etapas reais de construção. O segundo exige microfone local e checkpoints. Sem TTY, a saída continua simples e legível por agentes.

O console usa Rich e atualiza no máximo oito vezes por segundo. Mostra forma de onda resumida, energia real, texto previsto, p95 de inferência, perdas de fila e escores de eventos. Escores não são confiança calibrada. A interface é observadora: não controla a captura nem altera o estado do modelo. A animação não prova que o modelo reconhece fala corretamente.

> “Streaming puro” descreve processamento contínuo e filas limitadas; não garante menor latência ou mais precisão em qualquer computador. Meça antes/depois com a interface ligada. O runner documenta a execução em arquivo; o microfone físico exige um teste no seu equipamento.


Antes de seguir:
- [ ] Li docs/style.md.
- [ ] Usei docstrings de uma linha em inglês.
- [ ] Mantive a UI fora da callback.


## 20 · Um link para seu agente de IA

Roteiro simples, arquivos legíveis e documentação automática de cada execução.

A entrada /agents/ foi feita para você passar um único link. Ela contém HTML completo legível sem executar JavaScript, um guia em Markdown e um plano JSON. Não exige MCP, chave de API ou navegação pela interface visual. O agente pode obter o ZIP, conferir os arquivos e seguir a ordem das etapas.

```text
Leia https://sorputi.github.io/talkweave/agents/
Execute o roteiro em CPU, verifique as evidências e entregue o REPORT.md
com resultados reais, falhas de qualidade e itens não testados.
```

O que observar: Pedido pronto para enviar a um agente com ferramentas de leitura, arquivos e terminal. Um agente sem capacidade de executar deve declarar esse limite.


### Um comando, oito etapas verificáveis

```bash
python run.py
# Terminal interativo:
python run.py --ui
# Retomar conferindo hashes:
python run.py --resume
```

Arquivo completo: run.py

O que observar: As etapas cobrem referência NumPy, fixture, testes, codec, reconstrução, modelo duplex, geração livre e avaliação.

O runner trabalha em work/talkweave-demo e salva logs durante a execução. A cada etapa, registra comando, código de saída, duração, estado e hashes dos arquivos esperados. Ele interrompe na falha. A retomada só reutiliza evidências intactas e recusa misturar códigos de revisões diferentes na mesma pasta.


### O que o agente entrega

```text
work/talkweave-demo/REPORT.md
work/talkweave-demo/report.json
work/talkweave-demo/logs/
work/talkweave-demo/runs/
work/talkweave-demo/reply.wav.json
```

O que observar: O registro automático mostra execução. O agente deve complementar a análise de qualidade usando docs/report.md e os resultados efetivos.

AGENTS.md acompanha o ZIP e orienta o agente. docs/style.md define os padrões de organização, interfaces e streaming. docs/report.md é um modelo de relatório. llms.txt ajuda a descobrir a documentação; llms-full.txt reúne o guia e o manual; agents/plan.json informa passos, dependências, arquivos e hashes. Esses recursos são documentos, não um serviço que executa código remotamente.

> O acesso segue a configuração vigente do site. Se o agente encontrar uma tela de login e não tiver autorização, forneça o ZIP ou o guia baixado. A documentação para agentes não altera permissões de acesso.


Antes de seguir:
- [ ] Passei o link /agents/ ao agente.
- [ ] Conferi REPORT.md e logs.
- [ ] Separei sucesso de execução de qualidade de voz.


## 21 · Rust e outras linguagens

Contratos estáveis, regras para agentes e uma portabilidade que pode ser conferida.

Você pode pedir ao agente uma implementação em Rust, C++, Go, TypeScript ou outra linguagem. O site fornece regras e perfis de adaptação; o projeto de referência continua Python. Defina o escopo: runtime híbrido com modelo Python, inferência nativa ou treinamento e inferência nativos. Essas entregas exigem trabalhos e verificações diferentes.

```text
Leia https://sorputi.github.io/talkweave/agents/
Adapte para Rust com inferência nativa, seguindo a skill de portabilidade.
Preserve o streaming e entregue REPORT.md com testes e pendências.
```

O que observar: O agente encontra agents/port.md, agents/lang/rust.md e agents/skills/port/SKILL.md. Para outras linguagens, usa agents/lang/other.md.


### Portar comportamento, não só sintaxe

Preserve taxa de áudio, tamanho do quadro, relógio, ordem dos codebooks, vocabulário, estado da GRU e realimentação da saída anterior. Compare primeiro os bytes do protocolo; depois códigos do codec, reconstruções, estado e logits com pesos idênticos. Gate order e biases de GRUs podem variar entre backends. Compilar sem erro não prova que a rede é equivalente.

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```

O que observar: Execute dentro de rs/ numa máquina com Rust >=1.77. O starter contém trait Engine, Frame/Output e protocolo PCM; não implementa rede, treinamento, captura ou UI. Não havia compilador Rust neste ambiente; estes testes não foram executados aqui.

As convenções seguem a linguagem: Rust usa documentação /// em inglês, Go usa comentários iniciados pelo nome da função, C++ e TypeScript usam sua documentação idiomática. Mantenha pastas curtas e contratos explícitos. A UI de console e o logging continuam fora da callback; a escolha da linguagem não garante automaticamente maior velocidade ou precisão.

> A skill é um arquivo de instruções para o agente ler. Não é instalação automática nem um plugin universal. O relatório precisa separar compilação, paridade numérica, streaming, desempenho e qualidade de voz, marcando o que ficou não testado.


Antes de seguir:
- [ ] Escolhi o escopo híbrido ou nativo.
- [ ] Conferi contratos e vetores de teste.
- [ ] Documentei paridade e o que falta implementar.


## 22 · Referências e decisões de engenharia

Fontes primárias, leitura recomendada e o que cada uma sustenta.

Pesquisa consultada em 8 de novembro de 2026. As fontes abaixo são documentos dos autores, repositórios oficiais, artigos e documentação técnica. O código do laboratório foi escrito para este tutorial; não é um fork do livro nem uma implementação fiel de uma arquitetura comercial.

Veja as referências ao final.


### O que aproveitar dos livros

De Python Machine Learning: representação de atributos, referência simples, padronização sem vazamento, descida de gradiente, regularização e avaliação. Da documentação PyTorch: tensores, autograd e laço de otimização. Da literatura de áudio: codec neural, modelagem causal e fluxos de usuário/agente. A combinação ensina a construir e medir; não elimina o trabalho de coleta nem a diferença entre uma demonstração e um produto.


### Como manter o projeto reproduzível

Registre revisão do código, configuração, seed, ambiente e versão do dataset. Uma seed fixa ajuda a comparar mudanças locais, mas não garante resultados idênticos entre CPU, GPU e versões diferentes. Guarde métricas por sessão e amostras de áudio sem selecionar só as melhores. Ao trocar qualquer componente, reexecute os testes apropriados e compare o mesmo conjunto de validação.

Licença do código original: MIT, incluída no ZIP. Os projetos, livros, pesos e datasets citados têm seus próprios termos. O site não baixa pesos grandes, não usa o microfone automaticamente e não envia suas gravações a um serviço de treinamento.


Antes de seguir:
- [ ] Sei quais fontes sustentam as decisões.
- [ ] Guardei configurações e revisões.


## Referências

- [Python Machine Learning · código oficial](https://github.com/rasbt/python-machine-learning-book-3rd-edition) — Fundamentos: padronização, gradiente, regularização e avaliação.

- [PyTorch · treinamento e otimização](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — Autograd, loss e atualização de parâmetros.

- [Moshi · artigo de arquitetura](https://arxiv.org/abs/2410.00037) — Modelo de fala/texto com fluxos paralelos e codec neural.

- [PersonaPlex · pesquisa NVIDIA](https://research.nvidia.com/labs/adlr/personaplex/) — Full-duplex, controle de papel e voz.

- [Moshi · treinador oficial](https://github.com/kyutai-labs/moshi-finetune) — Referência de treinamento com dados de diálogo.

- [PersonaPlex · fine-tuning comunitário](https://github.com/tin-computer/personaplex-finetune) — Receita experimental; antigo endereço emotion-machine-org redireciona para este.

- [PersonaPlex · adaptação publicada em inglês](https://huggingface.co/demegire/personaplex-finetune-pharma) — Pesos, checkpoints e métricas de um ajuste de domínio; não comprova português.

- [PersonaPlex · discussão sobre português](https://huggingface.co/nvidia/personaplex-7b-v1/discussions/36) — Limitação linguística explicitada por integrante da NVIDIA.

- [Silero VAD · repositório oficial](https://github.com/snakers4/silero-vad) — Detecção de atividade de voz; não é reconhecedor de risadas.

- [faster-whisper · repositório oficial](https://github.com/SYSTRAN/faster-whisper) — ASR independente para comparação; não é o núcleo do nosso modelo.

- [AudioSet · ontologia de sons](https://research.google.com/audioset/ontology/index.html) — Taxonomia para sons humanos e ambientais.

- [PANNs · código dos autores](https://github.com/qiuqiangkong/audioset_tagging_cnn) — Classificação de eventos sonoros como referência avançada.

- [RNNoise · Xiph](https://github.com/xiph/rnnoise) — Supressão neural de ruído. Não substitui cancelamento de eco.

- [AudioWorklet · MDN](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet) — Processamento de áudio na thread de áudio do navegador.

- [CORAA · NILC](https://github.com/nilc-nlp/CORAA) — Fala em português para pesquisa; confira licenças e termos de cada componente.

- [LibriTTS · OpenSLR](https://www.openslr.org/60/) — Leitura em inglês, útil para áudio/voz; não contém diálogo full-duplex natural.
