# Otimizações ARM NEON — gemma4.c

Documento de referência para o kernel NEON int8 do projeto gemma4.c,
alvo: **Cortex-A72** (Raspberry Pi 4, 4 cores, base AdvSIMD sem dotprod/fp16).

---

## 1. Contexto

O gemma4.c faz inferência de Gemma 4 E2B em C puro, com pesos int8
(quantização por grupo de 64) e scales fp16. O gargalo dominante é o
`matmul_int8`: para cada token, 35 camadas × ~8 projeções lineares ×
1536×1536 (ou maiores no MLP).

O kernel NEON (`kernels_neon.c`) substitui o caminho escalar
(`kernels_pure.c`) e o AVX2 (`kernels.c`) em aarch64.

### Restrições do Cortex-A72

| Recurso | Disponível? |
|---------|------------|
| AdvSIMD (NEON base) | Sim |
| `SDOT`/`UDOT` (dotprod) | **Não** |
| FP16 arithmetic | **Não** |
| `vpadalq_s16` (SADALP) | Sim |
| `vpaddlq_s16` (SADDLPair) | Sim |
| `vpaddq_s32` (ADDP) | Sim |
| `vaddvq_f32` (horizontal add) | Sim |

Compilar com `-mcpu=cortex-a72` (não `-march=armv8-a`) para que o GCC
faça scheduling específico do pipeline A72 sem habilitar extensões
inexistentes.

---

## 2. Layout dos pesos (packed)

O `exporter.py` serializa os pesos int8 no layout:

```
[block][group][chunk][row][4]
```

- `block` = output_row / 16 (blocos de 16 saídas)
- `group` = input / 64 (grupos de 64 entradas, 24 grupos para width=1536)
- `chunk` = (input % 64) / 4 (16 chunks de 4 por grupo)
- `row` = output_row % 16
- `4` = 4 bytes contíguos (os 4 inputs do chunk)

Offset absoluto:

```
block * 16 * width + group * 16 * 64 + chunk * 64 + row * 4 + offset
```

**Por que esse layout é bom para NEON:**
Um load de 16 bytes (`vld1q_s8`) a partir de `wc + row*4` lê
exatamente 4 rows × 4 inputs contíguos — sem gather, sem reordenação.

---

## 3. Estratégia do matmul

### 3.1 Tiling: 2 input rows × 16 output rows

```
┌─────────────────────────────────────────────────┐
│  output_block (16 saídas)                       │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  group (64 inputs)                        │  │
│  │                                           │  │
│  │  chunk 0..15 (4 inputs cada)              │  │
│  │                                           │  │
│  │  4 loads de 16B (rows 0-3, 4-7, 8-11,    │  │
│  │  12-15) alimentam 2 tokens simultâneos    │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

- **2 input rows** compartilham os mesmos loads de peso (reuso de L1).
- **16 output rows** são processadas em 4 quartets (4×4) por chunk.
- Para `rows=1` (decode), a segunda row é duplicada (custo zero extra
  em memória, apenas ALU redundante).

### 3.2 Duplicação do input: `duplicate_4_s8`

```c
int8x16_t duplicate_4_s8(const int8_t *p) {
    uint32_t x;
    __builtin_memcpy(&x, p, sizeof(x));
    return vreinterpretq_s8_u32(vdupq_n_u32(x));
}
```

4 bytes de input → 16 bytes NEON:

```
x0 x1 x2 x3  →  x0 x1 x2 x3  x0 x1 x2 x3  x0 x1 x2 x3  x0 x1 x2 x3
```

O `vdupq_n_u32` replica o word em todas as 4 lanes de 32 bits.
O `__builtin_memcpy` evita problemas de strict-aliasing.

### 3.3 Dot product: `dot_4rows_4cols_acc`

```c
void dot_4rows_4cols_acc(int32x4_t *acc_lo, int32x4_t *acc_hi,
                         int8x16_t w, int8x16_t x) {
    int16x8_t lo = vmull_s8(vget_low_s8(w), vget_low_s8(x));
    int16x8_t hi = vmull_high_s8(w, x);
    *acc_lo = vpadalq_s16(*acc_lo, lo);
    *acc_hi = vpadalq_s16(*acc_hi, hi);
}
```

**Fluxo por chunk (4 inputs × 4 rows):**

```
w (16 bytes):          x (duplicado):
  w00 w01 w02 w03       x0  x1  x2  x3
  w10 w11 w12 w13       x0  x1  x2  x3
  w20 w21 w22 w23       x0  x1  x2  x3
  w30 w31 w32 w33       x0  x1  x2  x3
        │
        ▼
  vmull_s8 (lo):  w00*x0  w01*x1  w02*x2  w03*x3  w10*x0  w11*x1  w12*x2  w13*x3
  vmull_high_s8:  w20*x0  w21*x1  w22*x2  w23*x3  w30*x0  w31*x1  w32*x2  w33*x3
        │
        ▼
  vpadalq_s16:    (w00*x0+w01*x1)  (w02*x2+w03*x3)  (w10*x0+w11*x1)  (w12*x2+w13*x3)
                  (w20*x0+w21*x1)  (w22*x2+w23*x3)  (w30*x0+w31*x1)  (w32*x2+w33*x3)
```

Após os 16 chunks do grupo:

```c
int32x4_t acc = vpaddq_s32(acc_lo, acc_hi);
// lane 0 = dot(row0, x[0..63])
// lane 1 = dot(row1, x[0..63])
// lane 2 = dot(row2, x[0..63])
// lane 3 = dot(row3, x[0..63])
```

**Contagem de instruções por chunk (1 quartet):**

| Versão | Instruções |
|--------|-----------|
| Antiga (vmull + vpaddl + vpadd + vadd) | 6 |
| Atual (vmull + vpadalq_s16) | 4 |

A economia de 2 instruções × 4 quartets × 16 chunks × 24 grupos ×
2 tokens = **~7.500 instruções a menos por output_block**.

### 3.4 Acumulação e rescale

Por grupo de 64 inputs:

```
int32 dot (exato)
  → vcvtq_f32_s32 (conversão)
  → vmulq_n_f32 (× input_scale)
  → vmulq_f32 (× weight_scale)
  → vaddq_f32 (acumulador float)
```

O resultado float é acumulado ao longo dos 24 grupos. A ordem das
somas float pode diferir do caminho escalar, mas a diferença é
inferior ao roundoff (verificado: max_abs_diff = 0.0 no teste).

---

## 4. Cache de weight scales

### Problema

Cada `output_block` tem 16 rows × 24 groups = 384 scales fp16.
No prefill com `rows=512`, converter as scales dentro do loop de
input rows significava **512 conversões redundantes** dos mesmos
384 valores.

### Solução

Converter **uma única vez** antes do loop de input rows:

```c
float32x4_t scale_cache[groups * 4];  // 24×4 = 96 vetores de 4 floats

for (int group = 0; group < groups; group++) {
    const uint16_t *ws = weight->scales + ((output_block * groups + group) * 16);
    scale_cache[group * 4 + 0] = (float32x4_t) { fp16_to_f32(ws[0]), ... };
    // ...
}
```

Dentro do hot loop, apenas um load de registrador:

```c
float32x4_t ws0 = scale_cache[group * 4 + 0];
```

**Redução:** 512 × 384 = 196.608 conversões → 384 conversões (512×).

---

## 5. Conversão FP16 → FP32 (bitwise)

### Problema

A versão original usava `ldexpf()` (2 chamadas por valor, com
branching). No prefill, isso era chamado centenas de milhares de vezes.

### Solução

Conversão por manipulação de bits, sem FP16 hardware:

```c
static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        // subnormal ou zero
        if (mant == 0) bits = sign;
        else {
            int p = 31 - __builtin_clz(mant);
            bits = sign | ((uint32_t)(p + 103) << 23)
                 | ((mant << (23 - p)) & 0x007fffffu);
        }
    }
    else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
        if (mant) bits |= 0x00400000u;  // NaN
    }
    else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}
```

**Custo:** ~5-8 instruções ALU (vs. ~20-30 com `ldexpf`).
Validado para todos os 65.536 valores FP16 (zero, subnormais,
normais, inf, NaN).

---

## 6. Attention scores

```c
void attention_scores(...) {
    for (int key_index = 0; key_index < num_keys; key_index++) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 3 < head_dim; j += 4)
            acc = vmlaq_f32(acc, vld1q_f32(query + j), vld1q_f32(key + j));
        float sum = vaddvq_f32(acc);       // horizontal sum (4 lanes)
        for (; j < head_dim; j++)          // tail escalar
            sum += query[j] * key[j];
        scores[key_index] = sum;
    }
}
```

**Bug corrigido:** a versão anterior usava `vmlaq_f32` com `vdupq_n_f32`
no tail, o que adicionava o produto às 4 lanes e depois somava as 4
lanes → cada elemento do tail era contado 4×.

`vaddvq_f32` (ADDP + ADD) faz o horizontal sum em 2 instruções.

---

## 7. Weighted value sum

```c
for (int j = 0; j + 3 < head_dim; j += 4) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int key_index = 0; key_index < num_keys; key_index++) {
        acc = vmlaq_f32(acc, vdupq_n_f32(probabilities[key_index]),
                        vld1q_f32(value + j));
    }
    vst1q_f32(output + j, acc);
}
```

Broadcast do scalar `probabilities[key_index]` + load de 4 floats do
value cache. Sem bug de overlapping (o loop avança de 4 em 4).

---

## 8. Makefile

```makefile
CFLAGS_BASE = -std=c11 -O3 -Wall -Wextra -fopenmp

# NEON (aarch64):
CFLAGS = $(CFLAGS_BASE) -mcpu=cortex-a72
```

`-mcpu=cortex-a72` é preferível a `-march=armv8-a` porque:
- Habilita scheduling específico do pipeline A72 (out-of-order, 2-wide)
- Não habilita `dotprod` ou `fp16` (inexistentes no A72)
- Garante que o GCC não gere instruções de extensões ausentes

---

## 9. Verificação

### Teste de correção (`test_neon_matmul.c`)

- Gera pesos packed sintéticos no layout `[block][group][chunk][row][4]`
- Gera input int8 aleatório + scales
- Compara `matmul_int8` (NEON) vs. referência escalar
- **Resultado: max_abs_diff = 0.0** (match exato)

### Teste de inferência

- "What is the capital of France?" → "The capital of France is **Paris**."
- "Explain Rayleigh scattering" → resposta correta
- Saída idêntica ao build scalar (temperatura 0, greedy)

---

## 10. Performance (Cortex-A72, 4 cores, Pi 4)

| Métrica | Scalar (pure) | NEON v1 (gather) | NEON v2 (atual) |
|---------|--------------|-----------------|-----------------|
| pp512 | ~0.9 tok/s | 2.69 tok/s | **8.11 tok/s** |
| Decode (tg) | ~0.8 tok/s | ~1.2 tok/s | **1.51 tok/s** |
| Prefill 16 tok | ~1.3 tok/s | 3.8 tok/s | **9.9 tok/s** |

**Ganho total NEON v2 vs. scalar: ~9× no prefill, ~2× no decode.**

### Por que o decode melhora menos?

Com `rows=1`, o matmul é **memory-bound** (bandwidth dos 4.7 GB de
pesos int8). O tiling 2×16 duplica o trabalho ALU mas não reduz o
tráfego de memória. O ganho no decode vem de:
- `vpadalq_s16` (menos instruções por chunk)
- `fp16_to_f32` bitwise (menos stalls no rescale)
- Scale cache (menos conversões)

Para ganhos maiores no decode, seria necessário:
- Reduzir o tráfego de memória (ex.: quantização mais agressiva)
- Aproveitar melhor a L1 (weight blocking)
- Ou aceitar que 1.5 tok/s é o limite prático do A72 para 4.7 GB

---

## 10b. Kernel int4 (grupo 32)

Além do int8, o `kernels_neon.c` implementa `matmul_int4` para o formato
int4 (dois valores de 4 bits por byte, zero point 8, um scale fp16 por grupo
de 32 entradas). O layout dos pesos é:

```
data[block*16*(width/2) + group*16*16 + row*16 + byte]
scales[(block*groups + group)*16 + row]
```

Cada byte guarda duas entradas (nibble baixo = entrada par, nibble alto =
entrada ímpar). Uma linha de 16 bytes cobre um grupo inteiro de 32 entradas.

### Widening: `int4_row_to_s8`

Os 32 valores int4 de uma linha são ampliados para `int8x16` (nibble − 8) e
acumulados com o mesmo mecanismo `vmull_s8`/`vpadalq_s16` do caminho int8 —
o produto escalar inteiro é exato; só o passo de rescale muda (scale a cada
32 entradas em vez de 64):

```c
int8x16_t int4_row_to_s8(const uint8_t *row) {
    uint8x16_t packed = vld1q_u8(row);
    uint8x16_t low  = vandq_u8(packed, vdupq_n_u8(0x0f));
    uint8x16_t high = vrshlq_u8(packed, vdupq_n_u8(4));
    int8x16_t lo8 = vreinterpretq_s8_u8(vsubq_u8(low,  vdupq_n_u8(8)));
    int8x16_t hi8 = vreinterpretq_s8_u8(vsubq_u8(high, vdupq_n_u8(8)));
    return vcombine_s8(vget_low_s8(lo8), vget_low_s8(hi8));
}
```

### Estrutura

- **1×16 (decode)**: `matmul_int4_block_1x16` — 8 chunks de 4 por grupo,
  4 quartets de linhas, acumuladores `a?l`/`a?h` por quartet.
- **2×16 (prefill)**: `matmul_int4_block_2x16` — os loads de peso são
  compartilhados entre dois tokens; a linha ímpar de cauda é tratada com
  `has_row1 == 0`.
- **Scale cache**: como no int8, os 16 scales fp16 por (grupo, linha) são
  convertidos uma única vez para `float32x4_t` antes do loop de linhas.
- **`quantize_int4`** quantiza as ativações em grupos de 32 para casar com a
  granularidade dos scales int4.

O driver `matmul_int4` escolhe 1×16 para `rows == 1` e 2×16 caso contrário,
com `#pragma omp for` sobre os blocos de 16 saídas.

---

## 11. Otimizações futuras

| Prioridade | Item | Impacto esperado |
|-----------|------|-----------------|
| Alta | NEON no `quantize()` (vmax + vcvtnq_s32_f32) | ~10-20% no prefill |
| Média | Weight blocking para L1 (decode) | ~10-30% no decode |
| Média | Investigar hang do `--bench` com gen>0 | Usabilidade |
| Baixa | 4×16 tiling (se register pressure permitir) | ~10% no prefill |
| Baixa | `geglu` com NEON (vld1q + vget_lane) | ~5% |

---

## 12. Referências

- [Arm Cortex-A Comparison Table](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Cortex-A%20R%20M%20datasheets/Arm%20Cortex-A%20Comparison%20Table_v4.pdf)
- [Arm NEON Programming Quick Reference](https://developer.arm.com/community/arm-community-blogs/b/operating-systems-blog/posts/arm-neon-programming-quick-reference)
- [AArch64 NEON Intrinsics (ARM)](https://developer.arm.com/documentation/100952/latest/)
