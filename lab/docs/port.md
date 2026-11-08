# Adaptação por linguagem

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

## Dropout e paridade

A referência aceita dropout explícito de entrada e saída da GRU no treino. Na inferência, preserve o equivalente a eval(): nenhuma máscara aleatória. Não confunda dropout entre camadas de uma GRU com dropout recorrente; a referência tem uma camada. Teste paridade em modo de inferência e documente diferenças de regularização se portar o treino.
