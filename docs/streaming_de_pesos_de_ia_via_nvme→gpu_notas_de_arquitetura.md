# Streaming de Pesos de Modelos de IA via NVMe → GPU (Linux)

## Contexto e Motivação
Este documento consolida uma discussão técnica sobre a evolução do **AirLLM** e a viabilidade de um **novo runtime de inferência** focado em *streaming explícito de pesos*, inspirado em conceitos como **DirectStorage (Microsoft)** e **GPUDirect Storage (NVIDIA)**.

O problema central identificado é que, mesmo com otimizações atuais (AirLLM, DeepSpeed offload, etc.), o caminho:

```
NVMe → RAM → VRAM
```

continua sendo um gargalo estrutural. A proposta é tratar o **NVMe como parte da hierarquia de memória**, permitindo:

```
NVMe → DMA → VRAM
```

sem passagem intermediária pela RAM.

---

## Estado Atual (AirLLM)

### O que o AirLLM já resolve bem
- Separação física dos pesos por *layer*.
- Tamanhos previsíveis e determinísticos dos arquivos.
- Execução sequencial layer-by-layer.
- VRAM tratada como recurso escasso (cache temporário).

Esses pontos tornam o AirLLM **estruturalmente compatível** com GPUDirect Storage.

### Limitações estruturais
- Forte dependência de PyTorch (`torch.load`, `state_dict`).
- Pesos sempre transitam pela RAM.
- PyTorch assume controle total do allocator CUDA.

Essas limitações tornam difícil evoluir o AirLLM diretamente para um modelo de streaming NVMe→VRAM real.

---

## Insight Central

> O AirLLM já resolve o *problema lógico* do streaming de pesos.
>
> O que falta é resolver o *problema físico* da movimentação de dados.

Isso sugere que **um novo projeto** faz mais sentido do que um fork profundo.

---

## Direção Proposta: Novo Runtime de Inferência

### Princípios Fundamentais

1. **Pesos não passam pela RAM por padrão**  
   A RAM é usada apenas para metadata, scheduling e controle.

2. **Peso ≠ Tensor**  
   Pesos são blobs residentes (ou não). Tensores são views temporárias.

3. **NVMe faz parte da hierarquia de memória**  
   VRAM é um cache ativo; NVMe é a base persistente.

---

## Roadmap de 3 Fases

### 🟢 Fase 1 — MVP (Prova Técnica)
**Objetivo:** demonstrar que pesos podem ser carregados diretamente do NVMe para a VRAM usando GDS e consumidos pelo PyTorch.

- NVIDIA GPU + Linux
- GPUDirect Storage funcional
- 1 modelo suportado (ex: LLaMA-7B)
- Inferência batch=1
- Execução layer-by-layer
- Sem foco em performance máxima, apenas viabilidade

**Resultado esperado:**
> "Conseguimos inferir um modelo real sem que os pesos passem pela RAM."

---

### 🟡 Fase 2 — Runtime de Streaming
**Objetivo:** transformar o MVP em um runtime utilizável.

- Scheduler de residency de layers
- Prefetch de layer N+1 enquanto N executa
- Double-buffering em VRAM
- Cache configurável (quantos layers manter residentes)
- Formato de weights próprio (binário, alinhado, GDS-friendly)

**Resultado esperado:**
> Inferência estável, previsível e com latência controlada.

---

### 🔴 Fase 3 — Generalização e Integração
**Objetivo:** tornar o projeto relevante para o ecossistema.

- Suporte a múltiplos modelos
- Quantização (INT8 / INT4)
- Integração opcional com frameworks (PyTorch frontend)
- Abstração de backend (GDS hoje, outros no futuro)
- Documentação e exemplos

**Resultado esperado:**
> Runtime de referência para streaming de pesos em IA.

---

## MVP Detalhado — GDS + PyTorch Bridge

### Escopo do MVP

- **Não** suportar treino
- **Não** suportar batching
- **Não** suportar modelos arbitrários

Foco exclusivo:
> Provar o pipeline NVMe → VRAM → Compute

---

### Pipeline de Dados (MVP)

```
[layer_N.bin no NVMe]
        ↓ (GPUDirect Storage)
[CUDA buffer em VRAM]
        ↓ (view)
[Tensor PyTorch CUDA]
        ↓
[Execução da layer]
```

---

### Componentes do MVP

#### 1. Formato de Weights

- Arquivos `.bin` por layer
- Layout flat (sem pickle)
- Alinhamento mínimo (4KB ou maior)
- Metadata separada (`metadata.json`):
  - shape
  - dtype
  - ordem das layers

---

#### 2. Módulo GDS (C++/CUDA)

Responsabilidades:
- Alocar buffer CUDA
- Ler arquivo do NVMe direto para VRAM (GDS)
- Expor ponteiro + tamanho

Interface conceitual:
```
GdsBuffer load_layer(path, size)
```

---

#### 3. PyTorch Bridge

- Criar tensor CUDA a partir de memória externa
- Usar `from_blob` ou DLPack
- Garantir lifetime correto do buffer

Interface conceitual:
```
tensor = gds_tensor(path, shape, dtype)
```

---

#### 4. Scheduler Simples

- Executa layers em ordem fixa
- Libera buffer da layer anterior
- Opcional: prefetch síncrono da próxima layer

---

## Estrutura Inicial do Projeto

```
project/
 ├─ model/
 │   ├─ metadata.json
 │   ├─ layer_000.bin
 │   └─ layer_001.bin
 │
 ├─ runtime/
 │   ├─ scheduler.py
 │   ├─ residency.py
 │   ├─ torch_bridge.py
 │   └─ gds_io.cu
 │
 └─ examples/
     └─ inference_demo.py
```

---

## Posicionamento do Projeto

### O que este projeto **não** é

- Não é uma tentativa de reinventar frameworks de IA existentes.
- Não é uma alternativa direta ao PyTorch, TensorFlow ou vLLM.
- Não é focado inicialmente em treinamento em larga escala.

### O que este projeto **é**

> Um runtime de inferência experimental e pragmático que traz técnicas maduras de datacenter (GPUDirect Storage) para o **mercado consumidor, homelab e workstations Linux**.

Ele existe para resolver um problema que frameworks generalistas ainda não atacam bem:

- VRAM limitada em GPUs consumer
- NVMe extremamente rápido e subutilizado
- Latência sensível em inferência local
- Streaming fino de pesos, não de dados

### Público-alvo inicial

- Usuários avançados de Linux
- Homelabs
- Desenvolvedores que rodam LLMs localmente
- Pesquisadores interessados em runtimes de inferência
- Pessoas que hoje usam AirLLM, llama.cpp, vLLM em setups limitados por VRAM

---

## Por que isso faz sentido **agora**

Historicamente, tecnologias seguem este caminho:

```
Datacenter / HPC → Workstation → Consumidor
```

GPUDirect Storage já está:
- maduro
- testado em produção
- usado em treinamento e pipelines de dados

O que **não** existe ainda é sua aplicação em:

- inferência interativa
- streaming de pesos
- ambientes domésticos

Este projeto existe exatamente nesse intervalo.

---

## Narrativa do Projeto ("Why this exists")

> Modelos de IA estão crescendo mais rápido do que a VRAM.
>
> Enquanto isso, NVMe se tornou rápido o suficiente para atuar como uma extensão real da memória.
>
> O software ainda não acompanhou essa realidade.

Frameworks atuais assumem que:
- pesos devem caber inteiramente na VRAM
- ou, no máximo, passar pela RAM

Este projeto quebra essa suposição.

Ele trata:
- VRAM como cache ativo
- NVMe como base da hierarquia de memória
- pesos como recursos *residentes sob demanda*

Assim como engines gráficas aprenderam a fazer streaming de texturas, este runtime faz streaming de **pesos de modelos**.

---

## Visão de Longo Prazo

Se bem-sucedido, este projeto pode:

- inspirar mudanças em frameworks maiores
- servir de base para pesquisa acadêmica
- virar backend opcional para runtimes populares
- antecipar uma necessidade inevitável do ecossistema de IA

> Streaming explícito de pesos não é um truque.
>
> É uma consequência inevitável do crescimento dos modelos.

---

## Guia para IA Codificadora (Copilot / Claude / etc)

Este trecho serve como **orientação explícita para uma IA codificadora** entender o projeto **GdsLLM**, seus objetivos, restrições e o que deve ser implementado. Ele pode ser usado diretamente como contexto inicial (system / project prompt).

---

## Nome do Projeto

**GdsLLM**

> MVP / Prova de Conceito de um runtime de inferência de LLMs com *streaming explícito de pesos* usando **GPUDirect Storage (GDS)** no Linux.

---

## Objetivo Central (não negociável)

> **Demonstrar inferência de um LLM onde os pesos são carregados diretamente do NVMe para a VRAM, sem transitar pela RAM do sistema.**

Se os pesos passarem pela RAM, o objetivo do projeto não foi atendido.

---

## O Problema Que Estamos Resolvendo

Frameworks atuais assumem que:
- pesos do modelo devem residir inteiramente na VRAM, ou
- passar obrigatoriamente pela RAM antes de chegar à GPU

Isso cria gargalos graves em:
- GPUs consumer (VRAM limitada)
- inferência local / homelab
- modelos grandes

O GdsLLM trata:
- **VRAM como cache ativo**
- **NVMe como base da hierarquia de memória**

---

## Escopo do MVP (restrições claras)

A IA codificadora **não deve tentar generalizar demais**.

### O MVP DEVE:
- Rodar apenas em **Linux + NVIDIA GPU**
- Usar **GPUDirect Storage (cuFile / nvidia-fs)**
- Suportar **1 modelo fixo** (ex: LLaMA-7B)
- Executar inferência **batch = 1**
- Executar o modelo **layer-by-layer**
- Carregar **um layer por vez** do NVMe para a VRAM

### O MVP NÃO PRECISA:
- Treinar modelos
- Suportar batching
- Ser rápido ou otimizado
- Ter API estável
- Suportar múltiplos modelos
- Funcionar sem GDS

---

## Pipeline de Dados Esperado

```
[layer_X.bin no NVMe]
        ↓ (GPUDirect Storage)
[CUDA buffer em VRAM]
        ↓ (tensor view)
[Tensor CUDA válido no PyTorch]
        ↓
[Execução da layer]
```

**Proibido:**
```
NVMe → RAM → VRAM
```

---

## Formato de Weights (assumido)

- Um arquivo `.bin` por layer
- Conteúdo: pesos em layout flat (sem pickle)
- Alinhamento mínimo: 4KB
- Metadata separada (`metadata.json`) contendo:
  - shape
  - dtype
  - ordem das layers

A IA **não deve usar `torch.load()` para pesos**.

---

## Componentes Que Precisam Ser Implementados

### 1. Módulo GDS (baixo nível)

Responsável por:
- Inicializar cuFile
- Alocar buffer CUDA
- Ler arquivo do NVMe diretamente para VRAM

Interface conceitual:
```cpp
GdsBuffer load_layer(const char* path, size_t size);
```

---

### 2. Bridge PyTorch ↔ CUDA Memory

Responsável por:
- Criar um tensor CUDA a partir de memória externa
- Garantir lifetime correto do buffer

Interface conceitual:
```python
tensor = gds_tensor(path, shape, dtype)
```

---

### 3. Scheduler Simples

Responsável por:
- Executar layers em ordem fixa
- Garantir que apenas um layer esteja residente
- Liberar o buffer anterior após uso

Sem prefetch no MVP.

---

## Arquitetura Inicial Esperada

```
gdsllm/
 ├─ model/
 │   ├─ metadata.json
 │   ├─ layer_000.bin
 │   └─ layer_001.bin
 │
 ├─ runtime/
 │   ├─ gds_io.cu        # cuFile + CUDA
 │   ├─ torch_bridge.py # tensor from external memory
 │   ├─ scheduler.py    # execução layer-by-layer
 │   └─ __init__.py
 │
 └─ examples/
     └─ inference_demo.py
```

---

## Critério de Sucesso do MVP

O MVP é considerado bem-sucedido se:

- Um modelo real executa inferência corretamente
- Cada layer é carregado diretamente do NVMe para a VRAM
- Nenhum peso passa pela RAM do sistema
- O processo é reproduzível

Performance **não** é critério nesta fase.

---

## Mentalidade Esperada da IA Codificadora

- Priorizar **clareza arquitetural** sobre otimização
- Preferir código explícito a abstrações mágicas
- Assumir que este é um **runtime experimental**
- Tratar GDS como *first-class citizen*

> Este projeto não é um fork de AirLLM.
> 
> É um novo runtime inspirado por suas ideias.

---

## Considerações Finais

- O AirLLM continua sendo uma excelente referência conceitual.
- A proposta aqui é **um salto arquitetural**, não apenas uma otimização.
- O projeto começa nichado (NVIDIA + Linux), mas resolve um problema inevitável do futuro da IA.

> Modelos estão grandes demais para a VRAM.
>
> Streaming explícito de pesos não é opcional — é inevitável.

Este documento serve como **ponto de continuidade** para discussões futuras, design detalhado e implementação.

