# CLAUDE.md — chat-with-pdf

## Qué demuestra este proyecto

RAG (Retrieval-Augmented Generation): el usuario sube un PDF y hace preguntas en lenguaje natural. El sistema responde usando únicamente el contenido del documento, no el conocimiento general del LLM.

## Flujo técnico

```
PDF → extraer texto → trocear en chunks → embeddings → ChromaDB
                                              ↓
Usuario pregunta → embed de la pregunta → top-20 chunks
                                              ↓
                                      reranker → top-5
                                              ↓
                              top-5 + pregunta → Claude API → respuesta
```

## Módulos

| Archivo | Responsabilidad |
|---|---|
| `src/pdf_loader.py` | Extrae texto del PDF (solo) |
| `src/chunking.py` | 3 estrategias de chunking (token-based, recursive, semantic) |
| `src/embeddings.py` | `Embedder` Protocol + `LocalEmbedder` (sentence-transformers) + `VoyageEmbedder` (API) + factory `get_embedder()` |
| `src/vector_store.py` | ChromaDB in-process, depende de la abstracción `Embedder` |
| `src/reranker.py` | `Reranker` Protocol + `LocalReranker` (BGE) + `VoyageReranker` (rerank-2.5) + factory `get_reranker()` |
| `src/rag_chain.py` | `retrieve_and_answer()` pipeline end-to-end + `answer_question()` (prompt + Claude). XML delimiters + system prompt anti-injection. |
| `src/observability.py` | Logger SQLite multi-provider para llamadas a Claude / Voyage |
| `src/evaluation.py` | `recall@k`, MRR, runner del eval |
| `app.py` | UI en Streamlit: subir PDF, hacer preguntas |
| `scripts/run_evaluation.py` | Compara estrategias de chunking con el backend configurado |
| `scripts/run_retriever_experiments.py` | Compara 4 configs locales (requiere `--extra local`) |

## Decisiones técnicas

### Backends intercambiables (Voyage por default, local opcional)

- **Default: Voyage API.** Embedder `voyage-3-lite` (512 dim, $0.02/1M tokens). Reranker `rerank-2.5`. Imagen Docker resultante ~1.2 GB (sin PyTorch) — apta para Cloud Run / serverless. Sin el refactor, con torch + sentence-transformers la imagen era ~2.7 GB.

(Nota sobre tamaños: 1.2 GB es el "virtual size" de `docker images` incluyendo el OS base. El tamaño *comprimido* real — el que se transfiere al pushear/pullear a un registry — es ~250 MB, lo que importa para Cloud Run cold-start y storage de Artifact Registry.)
- **Local backend como extra `[local]`** en `pyproject.toml`. Instalación: `uv sync --extra local`. Trae sentence-transformers + torch (~1.2 GB más). Default: `BAAI/bge-base-en-v1.5` (embedder) + `BAAI/bge-reranker-base` (reranker).
- **Abstracción via `Protocol`, no herencia.** `Embedder` y `Reranker` son `typing.Protocol`. Las clases concretas no heredan de nada — solo cumplen el contrato. Structural typing / duck typing tipado.
- **Selección via env vars.** `EMBEDDING_BACKEND=voyage|local`, `RERANKER_BACKEND=voyage|local|off`. Si el factory recibe un valor inválido, levanta `ValueError`.
- **Lazy imports.** `sentence_transformers` se importa dentro del `__init__` de las clases locales para que la imagen liviana no falle en parse-time cuando torch no está instalado.

### Otras decisiones

- **ChromaDB embebido (in-process):** corre dentro de la misma app, sin servicio separado. Los datos no persisten entre reinicios — consciente para simplificar.
- **pypdf:** librería liviana para extracción. No soporta PDFs escaneados; fuera de scope.
- **Chunks de ~500 tokens con overlap de 50** como default — comparados contra otras estrategias.
- **Batching interno de Voyage embed:** la API tiene cap de 1000 textos por llamada. `VoyageEmbedder.embed()` trocea internamente; el caller no se entera. Un solo registro en `logs/llm_calls.db` agrega tokens y latencia de todos los sub-batches.
- **`input_type` asimétrico para Voyage:** `"document"` cuando se embeben chunks (al indexar), `"query"` cuando se embebe la pregunta. Voyage entrena asimétricamente — ignorar este parámetro pierde ~5-10% de recall. Modelos locales lo ignoran (bi-encoders simétricos).
- **Prompt injection mitigation:** los excerpts del PDF entran al prompt envueltos en `<document_excerpts><excerpt index="N">...</excerpt></document_excerpts>`, y el system prompt instruye a Claude a tratar ese contenido como dato, no instrucción. Mitiga PDFs maliciosos que intenten `Ignore previous instructions...`. Implementado en `src/rag_chain.py:answer_question`.

## Comparativa de chunking strategies

| Estrategia | Cómo trocea |
|---|---|
| `fixed_size` | 500 tokens con overlap 50 (baseline) |
| `recursive` | RecursiveCharacterTextSplitter de LangChain (split por párrafos → frases → chars) |
| `semantic` | Split por similitud entre frases consecutivas |

**Eval sets:** 3 documentos × 15-20 preguntas (`data/eval_questions*.json`). Cada pregunta tiene un `reference` que contiene la respuesta.

**Métricas:** `recall@k` (k=3, 5, 10), `MRR`.

Resultados completos en el README. TL;DR: recursive gana en académico-denso, token-based en prosa factual, semantic underperforma consistentemente.

## Reranker

Patrón "retrieval barato + rerank caro sobre subset" — estándar en RAG productivo:

- Retrieval inicial: top-20 chunks (no top-5).
- Reranker re-ordena esos 20 y devuelve top-5 al LLM.
- Cross-encoders (BGE local o rerank-2.5 API) hacen scoring jointly sobre cada par `(query, chunk)`.

**Finding crítico:** el reranker es la variable más confiable para mejorar retrieval. Multiplica `recall@3` por 2-3.6× consistentemente, más que cualquier swap de embedder.

## Observabilidad

Cada llamada a un proveedor externo (Claude, Voyage embed, Voyage rerank) se loggea en `logs/llm_calls.db` con tokens, latencia y costo. Esquema en `src/observability.py`. Voyage embeddings/rerankers tienen `output_tokens=0` (no aplica al modelo) — el costo se computa solo sobre input.

Pricing references en `_PRICING_PER_M_TOKENS`:
- Claude Sonnet 4.6: $3/$15 per 1M (in/out)
- voyage-3-lite / voyage-3 / voyage-3-large: $0.02 / $0.06 / $0.18 per 1M (input only)
- rerank-2.5-lite / rerank-2.5: $0.02 / $0.05 per 1M

## Limitaciones documentadas

- Preguntas multi-hop.
- PDFs con tablas complejas.
- Preguntas con negaciones o cuantificadores.
- Preguntas sobre el documento como un todo (RAG no es la herramienta correcta).
- **Cross-lingual:** los embedders por default (voyage-3-lite, BGE-base-en) son English-centric. Queries en español sobre PDFs en español degradan recall significativamente; el chunk correcto frecuentemente ni siquiera entra al top-20. Fix conocido: usar `voyage-multilingual-2` o `paraphrase-multilingual-mpnet-base-v2`.
- **Permisos de logs/ por Docker root user:** el contenedor corre como root y crea `logs/` con ownership root, que se filtra al host vía bind mount. Fix futuro: `USER appuser` al Dockerfile.

## Scope de esta versión

- Un PDF a la vez
- Sin historial de conversación
- Sin soporte para PDFs escaneados
- ChromaDB sin persistencia entre sesiones

## Cómo correr el proyecto

### Docker (producción, imagen liviana sin PyTorch)

```bash
cp .env.example .env
# editar .env con ANTHROPIC_API_KEY y VOYAGE_API_KEY
docker compose up --build
# http://localhost:8501
```

### Local con uv

```bash
# Solo backend Voyage:
uv sync
uv run streamlit run app.py

# Con backend local también:
uv sync --extra local
EMBEDDING_BACKEND=local RERANKER_BACKEND=local uv run streamlit run app.py
```

### Tests

```bash
uv run pytest  # 58 tests, ~3s. Usa FakeEmbedder + mocks de voyageai, no requiere [local].
```

### Eval

```bash
# Con el backend configurado en .env:
uv run python -m scripts.run_evaluation --eval data/eval_questions_falcao.json

# Experimento solo local:
uv sync --extra local
uv run python -m scripts.run_retriever_experiments --eval data/eval_questions_falcao.json
```

## Cuando trabajemos en este proyecto

- El refactor a backends pluggables ya está hecho. No tocar la abstracción `Protocol` sin discutirlo.
- Si vas a cambiar el modelo de embedding o reranker por default, **medir antes con `run_evaluation.py` sobre los 3 eval sets**. Tenemos data histórica para comparar.
- Las preguntas en español sobre contenido en español son **un caso conocido que falla**. Antes de "arreglar" un bug reportado, verificar si cae en esta categoría.
- Patrón para agregar nuevo proveedor (Cohere, OpenAI embeddings, etc.): nueva clase que cumpla `Embedder` protocol, agregar al factory, agregar pricing a `_PRICING_PER_M_TOKENS`, tests con mock de la API.
