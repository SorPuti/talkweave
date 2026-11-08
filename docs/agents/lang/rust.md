
[Contrato comum](https://sorputi.github.io/talkweave/agents/port.md) · [Skill](https://sorputi.github.io/talkweave/agents/skills/port/SKILL.md) · [Projeto completo](https://sorputi.github.io/talkweave/downloads/talkweave.zip)

# Perfil Rust

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
