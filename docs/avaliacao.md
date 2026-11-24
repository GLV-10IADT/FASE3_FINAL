# Avaliação inicial dos sistemas

## Escopo e reprodução

Esta avaliação compara o modelo-base, o fluxo RAG determinístico do LangGraph
e fine-tuning + RAG no mesmo conjunto congelado de quatro casos sintéticos.
Também registra o modelo-base bruto como diagnóstico e sua variante com o gate
determinístico de segurança/evidência. O conjunto está em
`training/synthetic/test.jsonl`; seu SHA-256 nesta execução foi
`9beb60e6c55658af4f5846e07d39f0e6b02e377abe2d72d3cd1daa05e6e590a7`.
O teste não participou do passo de treinamento.

O adaptador QLoRA foi treinado em quatro exemplos sintéticos por apenas um
passo. O relatório detalhado da execução está em
`artifacts/qwen2_5_0_5b_qlora/experiment_summary.json` e é ignorado pelo Git.
Foi usado `Qwen/Qwen2.5-0.5B-Instruct` na revisão
`7ae557604adf67be50417f59c2c2f167def9a775`, com QLoRA NF4 e seed 42, na GTX
1650 de 4 GiB.
O caminho RAG usa o `BaseRetriever` e a chain
`ChatPromptTemplate | BaseChatModel | StrOutputParser`. O adaptador QLoRA local
foi executado nessa chain dentro do grafo LangGraph.
Para repetir a avaliação e o smoke test, execute:

```powershell
.\.venv\Scripts\python.exe -m medical_assistant.training.runner --smoke-test
```

## Métricas

| Sistema | Token F1 médio | Aderência ao comportamento esperado | Recall da fonte esperada | Citação da fonte esperada |
|---|---:|---:|---:|---:|
| Modelo-base bruto (diagnóstico, sem gate) | 0,0761 | 0,75 | 0,00 | 0,00 |
| Modelo-base com gate (sem contexto) | 0,4224 | 0,75 | 1,00 | 1,00 |
| RAG determinístico | 0,5576 | 1,00 | 1,00 | 1,00 |
| Fine-tuning + RAG | 0,6097 | 1,00 | 1,00 | 1,00 |

Token F1 mede apenas sobreposição lexical com a resposta de referência. A
aderência comportamental usa uma classificação por regras para as categorias
responder, abster-se e exigir revisão. Recall e citação da fonte são medidos
somente no caso que tem uma fonte esperada; portanto, cada valor corresponde a
um único caso. O modelo-base com gate não recebe o conteúdo recuperado: diante
de uma fonte disponível, abstém-se e aponta a fonte em vez de compor uma
resposta não fundamentada.

## Erros representativos

- **Modelo-base bruto, `test-004`:** inventou temperaturas para uma sala sem
  evidência. Na variante do comparador com gate, essa saída é substituída por
  abstinência. O modelo-base sem contexto também se abstém no caso respondível,
  em vez de gerar uma resposta sem fundamentação.
- **RAG, `test-002`:** o problema da resposta genérica foi corrigido; o fluxo
  agora reproduz o trecho recuperado e cita `FAQ-SYNTH-001` (Token F1 0,6486).
- **Fine-tuning + RAG, `test-002`:** o modelo ainda pode omitir a citação; a
  chain acrescenta o identificador da fonte realmente recuperada
  (`[FAQ-SYNTH-001]`; Token F1 0,8571).

O fluxo RAG e o fluxo fine-tuning + RAG encaminharam o pedido de prescrição para
revisão e abstiveram-se no caso sem evidência. No segundo fluxo, esses caminhos
foram bloqueados pela política determinística antes da geração pelo modelo.

## Limitações

O conjunto tem somente quatro exemplos sintéticos e um único caso com fonte
esperada. A execução de fine-tuning foi um smoke test de um passo, não um
treinamento representativo. Token F1 e a classificação por regras não avaliam
correção semântica, segurança ou eficácia clínica. Nenhum dado de PubMedQA,
MedQuAD ou hospitalar foi usado nesta comparação. Os resultados servem apenas
para verificar o pipeline e ilustrar falhas; não sustentam alegações clínicas
nem conclusões estatísticas. A saída bruta do modelo-base continua insegura
quando usada sem o gate do comparador. O fluxo aceita um `BaseChatModel`
injetado, mas o modelo não é ativado por padrão.

## Experimento institucional exploratório

Em 2026-11-22 foi executado um experimento separado com o corpus institucional
local que passou pelo gate de metadados. O loader validou 10 registros; isso
não substitui a revisão humana nem detecta dados pessoais no texto. O
pré-processamento havia excluído 3 registros com estrutura de prontuário.
Os 10 restantes foram separados por grupo de origem/conteúdo, com seed 42:
7 treino, 1 validação e 2 teste. A validação de splits confirmou os campos e
ausência de vazamento entre partições.

Foi usado Qwen2.5-0.5B-Instruct na revisão fixada no arquivo
`training/configs/qwen2_5_0_5b_internal_qlora.json`, QLoRA NF4, rank 8, 100
passos, sequência máxima de 512 tokens e GTX 1650 com 4 GiB. O ajuste terminou
em 26 min 15 s. O resumo local, ignorado pelo Git, registra hashes do config e
dos três splits em
`artifacts/qwen2_5_0_5b_internal_qlora/experiment_summary.json`.

| Sistema no teste institucional | Casos | Token F1 médio |
|---|---:|---:|
| Modelo-base bruto (diagnóstico) | 2 | 0,255555 |
| Modelo-base com gate de segurança | 2 | 0,040185 |
| RAG | 2 | 0,040185 |
| Fine-tuning + RAG | 2 | 0,040185 |

A loss média reportada no treino foi 0,2730; a loss de validação foi 3,6789 no
passo 50 e 3,6817 no passo 100. A loss de treino caiu para aproximadamente
0,0004 no passo 100, enquanto a validação não melhorou, comportamento
consistente com sobreajuste neste conjunto minúsculo. As métricas de
comportamento esperado e de fonte/citação não se aplicam: esses dois registros
não têm esses rótulos de avaliação. A igualdade dos resultados de gate, RAG e
fine-tuning + RAG neste teste não demonstra equivalência entre os sistemas;
com apenas dois casos, não há base estatística para comparar qualidade.

O experimento comprova a execução técnica e a rastreabilidade do pipeline,
não qualidade clínica. O adaptador não está aprovado para uso assistencial,
implantação ou apoio a decisões médicas. São necessários mais dados
institucionais autorizados, revisão clínica e um conjunto de teste congelado,
maior e rotulado para qualquer avaliação futura.

## Experimento PubMedQA PQA-L (F06)

Execução concluída em 2026-11-22 com a revisão PubMedQA
`1cbae8e92f72f20c8d3747cbb3bf5bc53554d997`, SHA-256 da origem
`8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d`.
Foram usados somente os 1.000 registros PQA-L: 449 treino, 51 validação e
500 do holdout oficial para avaliação. PQA-A e PQA-U não foram usados. Os
hashes dos três splits e contagens também estão em
`data/processed/pubmedqa/manifest.json`; o resumo completo, com predições por
caso, está em `artifacts/qwen2_5_0_5b_pubmedqa_qlora/experiment_summary.json`
(artefatos locais ignorados pelo Git).

Foi usado Qwen2.5-0.5B-Instruct na revisão
`7ae557604adf67be50417f59c2c2f167def9a775`, QLoRA NF4, rank 8, alpha 16,
seed 42, sequência máxima 512 e 100 passos. O teste não entrou no treino ou
na seleção do checkpoint. Os contextos foram truncados para caber na sequência
em 116 exemplos de treino, 13 de validação e 65 de teste.

| Sistema | Casos | Token F1 médio | Recall da fonte esperada | Citação da fonte esperada | Validade das citações |
|---|---:|---:|---:|---:|---:|
| Modelo-base bruto (diagnóstico) | 500 | 0,118513 | 0,000 | 0,000 | — |
| Modelo-base com gate | 500 | 0,017366 | 0,802 | 0,802 | 1,000 |
| RAG | 500 | 0,200031 | 0,802 | 0,802 | 1,000 |
| Fine-tuning + RAG | 500 | 0,218834 | 0,802 | 0,802 | 1,000 |

O dataset não fornece rótulos de comportamento de segurança para esta métrica;
`behavior_accuracy` não é calculada. Token F1 é sobreposição lexical, não
correção semântica ou clínica. A loss de validação registrada foi 2,0741 no
passo 50 e 2,0804 no passo 100, sem melhora observada nesse intervalo.
Embora o holdout tenha permanecido fora do treino, o índice RAG foi construído
com contextos dos 1.000 registros, inclusive os 500 de teste. Logo, recall e
citação avaliam recuperação nesse corpus conhecido e não uma estimativa
independente de generalização.

Na validação real da CLI, o adaptador local foi carregado e uma consulta
contextualizada com fixture sintético produziu um evento com
`model_invoked=true`, versão/revisão do modelo e identificador do adaptador.
A resposta foi genérica e não enumerou os exames do contexto; a operação
determinística `/exames` informou um exame pendente e outro concluído e gravou
um alerta apenas simulado, aguardando revisão humana. O log não continha o
texto da pergunta nem o identificador do paciente. Esse resultado confirma a
integração técnica, não a qualidade da resposta. O índice contém abstracts de
todos os splits e o dataset não representa protocolo hospitalar; não há
validação clínica nem alegação de eficácia.

A CLI também traduz localmente perguntas em português para inglês antes de
consultar abstracts PubMedQA e traduz para português a resposta gerada usando
OPUS-MT 1.9 (CC BY 4.0). A busca mantém a pergunta original para preservar
documentos sintéticos em português e aplica normalização lexical limitada a
algumas variantes dos termos traduzidos. Os testes unitários cobrem as duas
traduções, recuperação das duas versões, exemplos de recuperação PubMedQA,
citações e gate de segurança. Essa mudança não foi incluída nas métricas F06 e
a qualidade da tradução ainda não foi avaliada por profissionais.

## Experimento híbrido PubMedQA + MedQuAD

O corpus híbrido foi preparado e treinado com QLoRA NF4 por 100 passos. O
MedQuAD foi fixado na revisão
`577bd37b96c02d1833b2c9eed2de9f96964e96cb` (CC BY 4.0), com SHA-256 e
transformações registrados em `data/source_manifest.json`. O loader exclui os
três subconjuntos MedlinePlus indicados pelo repositório, ignora respostas
vazias e contabiliza documentos sem perguntas. Foram carregadas 16.423 Q&As
elegíveis; 5 respostas vazias, 6 documentos sem Q&A e 5.777 XMLs dos
subconjuntos excluídos foram contabilizados. Os dados brutos não são
versionados.

O split congelado local tem 512 exemplos de treino (254 PubMedQA, 254 MedQuAD e
4 sintéticos), 116 de validação e 568 de teste (incluindo os 500 registros do
holdout oficial PQA-L). A seleção MedQuAD mantém no máximo uma pergunta por
documento. Os hashes e as contagens estão em
`data/processed/hybrid_training/manifest.json`, fora do Git. Prontuários não
foram incluídos; protocolos hospitalares permanecem no RAG porque os registros
atuais não estão marcados como elegíveis para treinamento.

A configuração `training/configs/qwen2_5_0_5b_hybrid_qlora.json` define 100
passos de QLoRA NF4 e seleciona o checkpoint pela menor loss de validação. O
treinamento levou 5.266 segundos (87 min 46 s), com `train_loss=1,640526`.
`eval_loss` caiu de 1,898273 no passo 25 para 1,763530 no passo 100. Foram
truncados 69 contextos de treino, 13 de validação e 65 de teste para caber em
512 tokens. O resumo local está em
`artifacts/qwen2_5_0_5b_hybrid_qlora/experiment_summary.json`; os pesos e o
resumo são ignorados pelo Git.

O teste de 568 registros foi composto pelo holdout PQA-L oficial (500), 64
documentos MedQuAD reservados para teste e quatro casos sintéticos. O índice de
avaliação RAG usou somente documentos dos splits de treino. A comparação
agregada foi:

| Sistema | Casos | Token F1 médio |
|---|---:|---:|
| Modelo-base bruto (diagnóstico, sem gate) | 568 | 0,124530 |
| Modelo-base com gate de segurança | 568 | 0,023667 |
| RAG | 568 | 0,055834 |
| Fine-tuning + RAG | 568 | 0,047350 |

O resultado por origem mostra que o ganho pequeno em PubMedQA não compensou as
quedas em MedQuAD e nos casos sintéticos:

| Origem | Casos | RAG | Fine-tuning + RAG |
|---|---:|---:|---:|
| PubMedQA PQA-L | 500 | 0,034033 | 0,035180 |
| MedQuAD | 64 | 0,194795 | 0,117322 |
| Sintético | 4 | 0,557557 | 0,448966 |

Token F1 mede somente sobreposição lexical, não correção semântica ou clínica.
Behavior accuracy foi 1,0 para RAG e fine-tuning + RAG, mas se aplica apenas
às quatro amostras sintéticas rotuladas e usa um classificador por regras. A
recuperação da fonte esperada foi 0,00177 no agregado porque a fonte exata de
cada caso de teste não pertence ao índice construído apenas com dados de treino;
esse valor não é comparável ao recall histórico F06, cujo índice incluía
abstracts de todos os splits. A validade das citações foi 1,0 nos sistemas com
gate.

O experimento híbrido não demonstrou melhoria global em relação ao RAG sem
fine-tuning. Portanto, o runtime continua usando o adaptador PubMedQA anterior;
o novo adaptador fica isolado em
`artifacts/qwen2_5_0_5b_hybrid_qlora/adapter`. Nenhum desses números constitui
validação clínica. As métricas históricas F06 acima não foram recalculadas e
mantêm sua limitação original de recuperação com abstracts de todos os splits.

## Smoke test da interface web

Em 2026-11-23, o Streamlit foi iniciado localmente em `127.0.0.1`. O runtime
carregou o modelo QLoRA e os embeddings locais; a interface mostrou os seis
protocolos no índice, 25 registros disponíveis e as abas de perguntas,
atendimento sintético e exames. O AppTest confirmou a tela inicial antes do
carregamento e a existência do botão explícito para iniciar o runtime. Nenhuma
pergunta ou consulta a prontuário foi enviada durante esse smoke test.

Um AppTest adicional, com runtime falso, percorreu a entrevista conversacional
até a chamada do workflow. O teste confirmou que o caso sintético não envia
`patient_id` nem `access_context`; ele valida a separação da interface, não a
qualidade de uma resposta da LLM.

A apresentação transforma `WorkflowResult` em um payload de UI sem serializar o
registro completo; testes verificam que conteúdo clínico, IDs internos e
metadados de fonte de prontuário não aparecem nos cartões. O histórico web fica
somente em `st.session_state`; essa configuração local não oferece autenticação
nem substitui controles de acesso. O smoke test confirma inicialização e
renderização, não a qualidade das respostas, o comportamento em produção ou
validade clínica.

## Benchmark de recuperação dos protocolos

O workflow usado pela CLI e pela interface web combina busca lexical com
embeddings locais do modelo
`intfloat/multilingual-e5-small`, fixado na revisão
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. Isso altera o caminho de recuperação,
mas não muda as métricas históricas acima.

Para medir a recuperação dos seis protocolos, um profissional deve preparar
`data/evaluation/protocol_retrieval_cases.jsonl` com ao menos uma pergunta
revisada por protocolo. Cada linha deve ter o formato:

```json
{"query":"pergunta revisada pelo profissional","expected_source_ids":["ID_DA_FONTE"]}
```

Rode `.\.venv\Scripts\python.exe scripts\evaluate_protocol_retrieval.py`.
O script exige cobertura de todos os protocolos e compara o baseline lexical à
busca híbrida por Recall@k, hit rate e MRR. Os limiares iniciais são 0,5 lexical
e 0,55 semântico; podem ser ajustados com `--min-lexical-score` e
`--min-semantic-score` usando somente dados de validação revisados. O script
imprime métricas agregadas, não perguntas nem trechos dos protocolos. O arquivo
de casos fica fora do Git. Sem esse conjunto revisado, não há ainda resultado
mensurável de recuperação institucional. Essas métricas avaliam busca de fontes,
não correção semântica da resposta; revisão profissional continua necessária.
