# Inference benchmark (plan §8.6, §10.4)

Prefill is one forward over batch x context tokens with no cache; latency is the mean over the measured repetitions (plan §8.6).
Decode is ms/token and tokens/s at each batch and cache length, the cache filled to the cache length before timing (plan §8.6).
Track `opt` runs opt mode, with the SDPA backend named in its source line enforced (plan §2.3 item 1). Track `ref` runs the reference implementation, which makes no SDPA call (plan §4.4).
KV bytes: the analytic size is per token, from the model's shapes; the measured size is the allocation growth of a cache holding the measured number of tokens, on CUDA. Plan §10.4 reads: "analytic and measured bytes/token (assert within 5%)."
Maximum batch: the largest batch whose decode step at a cache of 8,192 does not run out of memory (plan §8.6).
Weights are random, and lengths past the trained context of 1,024 measure speed only (plan §8.6).

## `mha`, opt mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `mha`, ref mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `gqa4`, opt mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `gqa4`, ref mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `mqa`, opt mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `mqa`, ref mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `mla192`, opt mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `mla192`, ref mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `gqa2`, opt mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `gqa2`, ref mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `mla96`, opt mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `mla96`, ref mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `swa256`, opt mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |

## `swa256`, ref mode

Source: no bench document given; measured cells are blank.

### Prefill

| context | batch | latency, ms | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |
| 1024 | — | — | — | — |
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

### Decode

| batch | cache | ms/token | tokens/s | peak allocated, MiB |
|---|---|---|---|---|
| 1 | 1024 | — | — | — |
| 1 | 4096 | — | — | — |
| 1 | 8192 | — | — | — |
| 8 | 1024 | — | — | — |
| 8 | 4096 | — | — | — |
| 8 | 8192 | — | — | — |
| 32 | 1024 | — | — | — |
| 32 | 4096 | — | — | — |
| 32 | 8192 | — | — | — |

### KV bytes and maximum batch

| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (plan §10.4) | max batch at cache 8k |
|---|---|---|---|---|
| — | — | — | — | — |
