# Análise Quantitativa do Trade-off entre Especialização e Generalização em LLMs via Fine-Tuning

Pipeline reprodutível para avaliar fine-tuning LoRA/QLoRA em Text-to-SQL usando Spider e regressão de capacidade usando MMLU.

## Estrutura

```text
configs/                  Configuração principal
custom_metrics/           Métrica ExecutionAccuracy em DeepEval
scripts/                  Preparação, treino, avaliação e sumarização
results/                  Saídas geradas
requirements.txt          Dependências fixadas
run_pipeline.sh           Pipeline completo
```

## Como rodar no Google Colab

Ative GPU em `Ambiente de execução > Alterar tipo de ambiente de execução > GPU`.

```bash
!git clone https://github.com/Clei21/Projeto2.git
%cd Projeto2
!pip install -q -r requirements.txt
!bash run_pipeline.sh
```

Se o download automático do Spider falhar, baixe o dataset pelo site oficial, envie o `.zip` para `data/spider.zip` e rode novamente:

```bash
!python scripts/prepare_spider.py
```

## Pipeline

1. Baixa e prepara o Spider
2. Converte `train_spider.json` e `dev.json` para JSONL
3. Avalia o modelo base no Spider dev split com Execution Accuracy
4. Avalia o modelo base no MMLU com 150 questões, 50 por categoria
5. Treina duas configurações LoRA/QLoRA
6. Avalia os modelos fine-tuned no Spider
7. Avalia os modelos fine-tuned no MMLU
8. Gera `results/summary.csv`

## Configuração base

Modelo padrão:

```yaml
Qwen/Qwen2.5-3B-Instruct
```

Configuração LoRA padrão:

```yaml
r: 16
alpha: 32
dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

Duas configurações experimentais:

```bash
learning_rate=0.0002, epochs=1
learning_rate=0.0001, epochs=1
```

## Resultados

Arquivos principais:

```text
results/spider_baseline.jsonl
results/spider_lora_lr2e-4_ep1.jsonl
results/spider_lora_lr1e-4_ep1.jsonl
results/mmlu_baseline.json
results/mmlu_lora_lr2e-4_ep1.json
results/mmlu_lora_lr1e-4_ep1.json
results/summary.csv
```

## Observações importantes

Use `temperature=0` e `do_sample=False` para avaliação determinística. A seed é fixada em `configs/default.yaml`.

Para Colab gratuito, mantenha `max_train_samples` e `max_eval_samples` reduzidos no primeiro teste. Depois aumente gradualmente.
