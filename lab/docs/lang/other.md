# Perfil para outra linguagem

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
