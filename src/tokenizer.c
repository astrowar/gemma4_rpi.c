#include "gemma4.h"

int compare_lookup_keys(const void *key, const void *entry) {
    return memcmp(key, ((const LookupEntry *)entry)->key, 8);
}

// Repeatedly applies the highest-priority learned merge until no adjacent token pair matches.
int apply_bpe_merges(const Tokenizer *tokenizer, int *tokens, int count) {
    for (;;) {
        const LookupEntry *best_merge = NULL;
        int position = -1;
        for (int i = 0; i + 1 < count; i++) {
            const LookupEntry *merge = bsearch(tokens + i, tokenizer->merges, tokenizer->merge_count, sizeof(LookupEntry), compare_lookup_keys);
            if (merge && (!best_merge || merge->rank < best_merge->rank)) { best_merge = merge; position = i; }
        }
        if (!best_merge) return count;

        tokens[position] = best_merge->result;
        memmove(tokens + position + 1, tokens + position + 2, (count - position - 2) * sizeof(*tokens));
        count--;
    }
}

// Converts the three prompt segments from UTF-8 into vocabulary pieces, falls back to byte tokens when needed, applies BPE, and prepends <bos>.
int tokenize(const Tokenizer *tokenizer, const char *segments[3], int *tokens, int capacity) {
    int count = 1;

    for (int segment = 0; segment < 3; segment++)
        for (const char *cursor = segments[segment]; *cursor;) {
            if (count >= capacity) return -1;
            int special = -1;
            if (*cursor == '<')
                for (int i = 0; i < tokenizer->special_count && special < 0; i++) {
                    int length = (int)strlen(tokenizer->specials[i].token);
                    if (!strncmp(cursor, tokenizer->specials[i].token, length)) { special = tokenizer->specials[i].id; cursor += length; }
                }
            if (special >= 0) { tokens[count++] = special; continue; }
            char piece[8] = {0};
            if (*cursor == ' ') { memcpy(piece, "\xE2\x96\x81", 3); cursor++; } // SentencePiece represents spaces with U+2581.
            else {
                piece[0] = *cursor++;
                if ((piece[0] & 0xC0) == 0xC0)
                    for (int i = 1; i < 4 && (*cursor & 0xC0) == 0x80; i++) piece[i] = *cursor++;
            }
            const LookupEntry *entry = bsearch(piece, tokenizer->encode_vocab, tokenizer->encode_vocab_count,
                                      sizeof(LookupEntry), compare_lookup_keys);
            if (entry) { tokens[count++] = entry->result; continue; }

            for (const unsigned char *byte = (const unsigned char *)piece; *byte; byte++) {
                if (count >= capacity) return -1;
                tokens[count++] = 238 + *byte; // Byte tokens occupy IDs 238 through 493.
            }
        }

    count = 1 + apply_bpe_merges(tokenizer, tokens + 1, count - 1);
    tokens[0] = 2; // Token 2 is <bos>.
    return count;
}

const char *token_text(const Tokenizer *tokenizer, int token) {
    return token >= 0 && token < VOCAB_SIZE ? tokenizer->decoded_tokens[token] : "";
}
