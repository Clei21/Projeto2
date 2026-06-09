# Análise Quantitativa do Trade-off entre Especialização e Generalização em LLMs via Fine-Tuning

Trabalho Prático 2 — ICC220 / PPGINF528 (UFAM, 2026/01)

Este repositório implementa o pipeline completo do experimento: fine-tuning de um
LLM (Qwen2.5-3B-Instruct) para Text-to-SQL com QLoRA no Spider, avaliação por
Execution Accuracy (métrica customizada em DeepEval) e medição da regressão de
capacidade geral em uma suíte de 150 questões do MMLU.

## Estrutura do repositório

```
.
├── configs/
│   ├── lora_a.yaml            # configuração A (lr = 2e-4)
│   └── lora_b.yaml            # configuração B (lr = 2e-5)
├── custom_metrics/
│   └── execution_accuracy.py  # métrica customizada (Fase 1)
├── scripts/
│   ├── spider_common.py       # serialização de esquema, prompt, seeds
│   ├── prepare_spider.py      # pré-processamento do Spider
│   ├── train_lora.py          # fine-tuning QLoRA (Fase 3)
│   ├── generate_predictions.py# inferência no dev split (Fases 2 e 4)
│   ├── build_mmlu_suite.py    # suíte de 150 questões do MMLU
│   ├── evaluate_mmlu.py       # avaliação 5-shot no MMLU (Fase 5)
│   └── analyze_tradeoff.py    # consolidação dos resultados
├── tests/
│   └── test_spider_eval.py    # integração da métrica com pytest (Fase 4.1)
├── report/
│   └── relatorio.tex          # template IEEE do relatório
└── requirements.txt
```

## 1. Ambiente

Hardware de referência: Google Colab (tier gratuito), GPU NVIDIA T4 (16 GB VRAM).
O treinamento usa QLoRA (quantização NF4 em 4 bits), que cabe confortavelmente
na T4 para modelos de 3B.

```bash
git clone <url-do-repositorio>
cd <repositorio>
pip install -r requirements.txt
```

No Colab, monte o Google Drive para persistir checkpoints entre sessões:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Obtenção dos dados

### Spider

Baixe o dataset no site oficial (https://yale-lily.github.io/spider) e extraia
na raiz do repositório. A estrutura esperada é:

```
spider/
├── train_spider.json
├── dev.json
├── tables.json
└── database/            # um subdiretório .sqlite por banco
```

### Pré-processamento

```bash
python scripts/prepare_spider.py --spider_dir spider --out_dir data
```

Gera `data/train.jsonl` (formato de chat para o SFTTrainer), `data/dev.jsonl`
(esquema + pergunta + SQL gold) e `data/few_shot.json` (os 3 exemplos fixos do
prompt, amostrados do training split com seed 42).

### Suíte MMLU

```bash
python scripts/build_mmlu_suite.py --out data/mmlu_suite.json
```

Amostra (seed 42) 50 questões de cada subcategoria: `college_computer_science`
(STEM), `philosophy` (Humanidades) e `high_school_macroeconomics` (Ciências
Sociais), além dos 5 exemplos do dev split de cada disciplina usados como
contexto 5-shot idêntico em todas as avaliações.

## 3. Baseline (Fase 2)

```bash
python scripts/generate_predictions.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --dev_file data/dev.jsonl \
  --few_shot data/few_shot.json \
  --out results/predictions_baseline.jsonl
```

A geração é determinística (greedy, `do_sample=False`). Use `--limit 50` para
um teste rápido antes da execução completa (~1034 exemplos, 2–4 h na T4).

Avaliação com a métrica de Execution Accuracy via pytest:

```bash
PREDICTIONS=results/predictions_baseline.jsonl \
SPIDER_DB_DIR=spider/database \
pytest tests/test_spider_eval.py -s
```

O relatório detalhado (acurácia agregada + resultado por exemplo) é salvo em
`results/eval_predictions_baseline.json`.

## 4. Fine-tuning (Fase 3)

Duas configurações de hiperparâmetros são obrigatórias; aqui variamos a taxa de
aprendizado em uma ordem de grandeza:

```bash
python scripts/train_lora.py --config configs/lora_a.yaml   # lr = 2e-4
python scripts/train_lora.py --config configs/lora_b.yaml   # lr = 2e-5
```

Configuração LoRA comum às duas execuções: r=16, alpha=32, dropout=0.05,
target_modules = q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
Tempo aproximado na T4: 4–6 h por configuração (2 épocas, ~7000 exemplos).
Os adaptadores são salvos em `outputs/lora_a` e `outputs/lora_b`.

## 5. Avaliação dos modelos fine-tuned (Fase 4)

Mesmo procedimento do baseline, apenas adicionando `--adapter`:

```bash
python scripts/generate_predictions.py \
  --model_name Qwen/Qwen2.5-3B-Instruct --adapter outputs/lora_a \
  --dev_file data/dev.jsonl --few_shot data/few_shot.json \
  --out results/predictions_lora_a.jsonl

PREDICTIONS=results/predictions_lora_a.jsonl \
SPIDER_DB_DIR=spider/database \
pytest tests/test_spider_eval.py -s
```

Repita para `lora_b`.

## 6. Regressão de capacidade no MMLU (Fase 5)

```bash
python scripts/evaluate_mmlu.py --model_name Qwen/Qwen2.5-3B-Instruct \
  --suite data/mmlu_suite.json --out results/mmlu_baseline.json

python scripts/evaluate_mmlu.py --model_name Qwen/Qwen2.5-3B-Instruct \
  --adapter outputs/lora_a --suite data/mmlu_suite.json \
  --out results/mmlu_lora_a.json
```

A resposta é determinada pela comparação dos logits dos tokens A/B/C/D na
posição de resposta (procedimento determinístico, padrão para múltipla escolha).

## 7. Consolidação do trade-off

```bash
python scripts/analyze_tradeoff.py \
  --spider_baseline results/eval_predictions_baseline.json \
  --spider_finetuned results/eval_predictions_lora_a.json \
  --mmlu_baseline results/mmlu_baseline.json \
  --mmlu_finetuned results/mmlu_lora_a.json \
  --out results/tradeoff_lora_a.json
```

Imprime e salva o ganho na tarefa-alvo e a variação percentual no MMLU,
agregada e por categoria.

## Reprodutibilidade

- Seed global fixa (42) em todas as operações estocásticas: amostragem dos
  exemplos few-shot, amostragem das questões do MMLU, inicialização do
  treinamento (`SFTConfig(seed=42)`) e bibliotecas (random, numpy, torch).
- Decodificação determinística (greedy) em todas as gerações de avaliação.
- Versões de todas as dependências fixadas em `requirements.txt`.
- Os mesmos 5 exemplos de contexto do MMLU e o mesmo prompt few-shot do Spider
  são usados para o baseline e para todos os modelos fine-tuned.
