# Análise Quantitativa do Trade-off entre Especialização e Generalização em LLMs via Fine-Tuning

Pipeline experimental para fine-tuning de um LLM na tarefa Text-to-SQL (benchmark Spider) com medição simultânea do ganho de especialização e da regressão de capacidade geral (MMLU).

**Disciplina:** ICC220 / PPGINF528 — UFAM, 2026/01

## Estrutura do repositório

```
text2sql_finetuning/
├── custom_metrics/
│   ├── __init__.py
│   └── execution_accuracy.py      # Métrica customizada (DeepEval BaseMetric)
├── scripts/
│   ├── config.py                  # Caminhos, seeds, modelo base, subcategorias MMLU
│   ├── download_data.py           # Auxiliar para descompactar o Spider
│   ├── preprocess_spider.py       # Spider train → formato chat (jsonl)
│   ├── spider_utils.py            # Serialização de schema e construção de prompts
│   ├── model_utils.py             # Carregamento de modelo/tokenizer e geração greedy
│   ├── train.py                   # Fine-tuning LoRA/QLoRA via TRL SFTTrainer
│   ├── evaluate_text2sql.py       # Avaliação Text-to-SQL (baseline e fine-tuned)
│   ├── evaluate_mmlu.py           # Avaliação MMLU 5-shot por categoria
│   ├── analyze_regression.py      # Consolidação dos deltas de capacidade
│   ├── error_analysis.py          # Extrai exemplos de falha para o relatório
│   └── test_evaluation.py         # Teste pytest integrando a métrica (Fase 4.1)
├── configs/
│   ├── config_a.yaml              # Hiperparâmetros: 1 época, lr=2e-4
│   └── config_b.yaml              # Hiperparâmetros: 3 épocas, lr=5e-5
├── notebooks/
│   └── run_in_colab.ipynb         # Notebook orquestrador para Google Colab
├── data/                          # Spider (não versionado — ver instruções abaixo)
├── results/                       # Saídas JSON das avaliações
├── requirements.txt
└── run_pipeline.sh
```

## Configuração do ambiente

**Ambiente alvo:** Google Colab (T4, 16 GB VRAM) com QLoRA 4-bit. GPUs maiores (A100, L4) permitem o uso de modelos 7-8B.

```bash
pip install -r requirements.txt
export PYTHONPATH=.
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct   # ou meta-llama/Llama-3.2-3B-Instruct
export SEED=42
```

**Modelos suportados (3-4B, T4):**
- `Qwen/Qwen2.5-3B-Instruct` *(padrão)*
- `meta-llama/Llama-3.2-3B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`

## Dados

### Spider

Baixe o arquivo oficial `spider_data` (site do benchmark ou repositório Yale) e organize-o assim:

```
data/spider/train_spider.json
data/spider/dev.json
data/spider/tables.json
data/spider/database/<db_id>/<db_id>.sqlite
```

Atalho para descompactar um zip já baixado:

```bash
python scripts/download_data.py --zip_path /caminho/spider_data.zip
```

### MMLU

O MMLU é baixado automaticamente do Hugging Face Hub (`cais/mmlu`) durante a avaliação. Subcategorias usadas:

| Domínio          | Subcategoria                   | Questões |
|------------------|--------------------------------|----------|
| STEM             | `college_computer_science`     | 50       |
| Humanidades      | `philosophy`                   | 50       |
| Ciências Sociais | `high_school_macroeconomics`   | 50       |

## Reprodução completa

```bash
bash run_pipeline.sh
```

### Passo a passo

```bash
# 1. Pré-processamento
python scripts/preprocess_spider.py

# 2. Baseline Text-to-SQL (Fase 2)
python scripts/evaluate_text2sql.py --run_name baseline

# 3. Baseline MMLU (Fase 5 — modelo base)
python scripts/evaluate_mmlu.py --run_name baseline

# 4. Fine-tuning — duas configurações (Fase 3)
python scripts/train.py --config configs/config_a.yaml --output_dir adapters/config_a
python scripts/train.py --config configs/config_b.yaml --output_dir adapters/config_b

# 5. Avaliação Text-to-SQL dos modelos fine-tuned (Fase 4)
python scripts/evaluate_text2sql.py --adapter_path adapters/config_a --run_name finetuned_a
python scripts/evaluate_text2sql.py --adapter_path adapters/config_b --run_name finetuned_b

# 6. Avaliação MMLU dos modelos fine-tuned (Fase 5)
python scripts/evaluate_mmlu.py --adapter_path adapters/config_a --run_name finetuned_a
python scripts/evaluate_mmlu.py --adapter_path adapters/config_b --run_name finetuned_b

# 7. Gate de avaliação via pytest (Fase 4.1)
EVAL_RUN_NAME=finetuned_a python -m pytest scripts/test_evaluation.py -v

# 8. Análise de regressão (Fase 5.3)
python scripts/analyze_regression.py --baseline_run baseline --finetuned_runs finetuned_a finetuned_b

# 9. Análise de erros para o relatório (2-3 exemplos de falha)
python scripts/error_analysis.py --run_name finetuned_a --n 3
```

O resumo consolidado fica em `results/summary.json`. As análises de erro ficam em `results/error_analysis_<run>.json`.

## Configuração LoRA

| Parâmetro         | Config A          | Config B          |
|-------------------|-------------------|-------------------|
| Técnica           | QLoRA (4-bit NF4) | QLoRA (4-bit NF4) |
| Rank (r)          | 16                | 16                |
| Alpha (α)         | 32                | 32                |
| Dropout           | 0.05              | 0.05              |
| Target modules    | q, k, v, o_proj   | q, k, v, o_proj   |
| Épocas            | 1                 | 3                 |
| Learning rate     | 2e-4              | 5e-5              |
| LR scheduler      | cosine            | cosine            |
| Batch size        | 4                 | 4                 |
| Grad. accum.      | 4 (eff. batch 16) | 4 (eff. batch 16) |
| Optimizador       | paged_adamw_8bit  | paged_adamw_8bit  |
| Max seq. length   | 2048              | 2048              |

## Métrica de Execution Accuracy

Implementada em [custom_metrics/execution_accuracy.py](custom_metrics/execution_accuracy.py), herdando de `deepeval.metrics.BaseMetric`.

Fluxo do método `measure(test_case)`:
1. Extrai o `db_id` do metadata do `LLMTestCase`
2. Extrai a consulta SQL da saída bruta (remove blocos markdown, texto explicativo, prefixo "SQL:")
3. Conecta ao banco SQLite correspondente em `data/spider/database/`
4. Executa a consulta predita e a gold em transações protegidas por `try-except`
5. Compara os conjuntos de resultados: insensível à ordem, **exceto** quando a gold contém `ORDER BY`
6. Retorna `1.0` (sucesso) ou `0.0` (falha)

A mesma instância da métrica é aplicada de forma idêntica ao baseline e a todos os modelos fine-tuned, garantindo comparabilidade total.

## Reprodutibilidade

- Sementes fixadas em `scripts/config.py` → `set_global_seed(42)` para Python, NumPy e PyTorch
- Toda geração de texto usa decodificação **greedy** (`do_sample=False`, `temperature=None`), garantindo saídas determinísticas
- Versões de todas as dependências fixadas em `requirements.txt`
- Os mesmos 3 exemplos few-shot (selecionados deterministicamente do training split) são usados em todas as avaliações Text-to-SQL
- Os mesmos 5 exemplos MMLU (split `dev`) são usados para todos os modelos em cada subcategoria

## Hardware

Documente no relatório o modelo de GPU e a VRAM utilizados. Exemplo para T4:

```
GPU: Tesla T4
VRAM: 15.8 GB
Técnica: QLoRA 4-bit (NF4, double quantization)
```

## Saídas geradas

| Arquivo                              | Conteúdo                                          |
|--------------------------------------|---------------------------------------------------|
| `results/text2sql_baseline.json`     | Acurácia baseline + todas as predições            |
| `results/text2sql_finetuned_a.json`  | Acurácia config A + todas as predições            |
| `results/text2sql_finetuned_b.json`  | Acurácia config B + todas as predições            |
| `results/mmlu_baseline.json`         | Acurácia MMLU baseline por categoria              |
| `results/mmlu_finetuned_a.json`      | Acurácia MMLU config A por categoria              |
| `results/mmlu_finetuned_b.json`      | Acurácia MMLU config B por categoria              |
| `results/summary.json`               | Deltas percentuais consolidados (relatório)       |
| `results/error_analysis_*.json`      | Exemplos de falha para análise qualitativa        |
