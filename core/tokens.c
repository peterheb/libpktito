/* Tokens to text, as NeMo writes a transcript.
 *
 * NeMo's text is SentencePiece's decoding of the token ids
 * (decode_tokens_to_str: sp.decode_pieces) with one change made after it
 * (decode_tokens_to_str_with_strip_punctuation): a whitespace character
 * just before a punctuation mark is taken away, the marks being every
 * punctuation character in the vocabulary's ordinary pieces (the converter
 * collects them as NeMo does and the file carries them). For Parakeet
 * that is ! % ' , - . / : ? and the Spanish openers ¡ ¿.
 *
 * The decoding is sentencepiece's Decode for the models Parakeet ships
 * (add_dummy_prefix, no extra-whitespace removal, no byte pieces, which the
 * converter checks): a control piece writes nothing; the unknown piece
 * writes SentencePiece's default surface, " ⁇ "; any other piece writes
 * its text with U+2581 made a space, except that the first piece to start
 * with U+2581 while nothing has been written has that one consumed.
 * tools/pktito_tools/spm.py has the same in Python, and tests/tokcheck.py
 * holds the two to each other and to sentencepiece itself.
 */
#include "model.h"
#include "pkf.h"

#include <string.h>

static const char SPACE[3] = { (char)0xE2, (char)0x96, (char)0x81 };   /* U+2581 */
static const char UNK[] = " \xE2\x81\x87 ";                         /* " ⁇ " */

typedef struct {
    char  *out;
    size_t cap, len;
    int    last_ws;          /* the last character written is whitespace */
    const pk_model *m;
} writer;

static size_t utf8_len(unsigned char c) {
    return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
}

static int is_ws(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

/* the character at s (n bytes long) is one NeMo strips the space before */
static int is_punct(const pk_model *m, const char *s, size_t n) {
    for (uint32_t i = 0; i < m->punct_len; i += (uint32_t)utf8_len((unsigned char)m->punct[i]))
        if (utf8_len((unsigned char)m->punct[i]) == n && !memcmp(m->punct + i, s, n))
            return 1;
    return 0;
}

static void put_char(writer *w, const char *s, size_t n) {
    if (w->last_ws && is_punct(w->m, s, n))
        w->len--;                                   /* (\s)(punct) -> punct */
    for (size_t i = 0; i < n; i++, w->len++)
        if (w->len < w->cap)
            w->out[w->len] = s[i];
    w->last_ws = n == 1 && is_ws((unsigned char)s[0]);
}

/* text of n bytes, U+2581 written as a space */
static void put_text(writer *w, const char *s, size_t n) {
    for (size_t i = 0; i < n;) {
        size_t k = utf8_len((unsigned char)s[i]);
        if (k == 3 && !memcmp(s + i, SPACE, 3))
            put_char(w, " ", 1);
        else
            put_char(w, s + i, k);
        i += k;
    }
}

size_t pk_text(const pk_model *m, const pk_tok *t, size_t n, char *out, size_t cap) {
    writer w = { out, cap, 0, 0, m };
    int consumed = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t id = (uint32_t)t[i].id;
        if (id >= m->n_pieces)
            continue;                               /* the blank, never emitted */
        uint8_t type = m->piece_type[id];
        const char *s = m->piece_text + m->piece_off[id];
        size_t len = m->piece_off[id + 1] - m->piece_off[id];
        int bos = !consumed && w.len == 0;
        consumed = 0;
        if (type == PK_PIECE_CONTROL)
            continue;
        if (type == PK_PIECE_UNKNOWN) {
            put_text(&w, UNK, sizeof UNK - 1);
            continue;
        }
        if (bos && len >= 3 && !memcmp(s, SPACE, 3)) {
            s += 3;
            len -= 3;
            consumed = 1;
        }
        put_text(&w, s, len);
    }
    if (cap) {
        size_t end = w.len < cap ? w.len : cap - 1;
        if (end < w.len) {
            /* cut at a character's start: back to the lead byte before the
             * cut, and before it too if its character does not fit */
            size_t s = end;
            while (s > 0 && ((unsigned char)out[s - 1] & 0xC0) == 0x80) s--;
            if (s > 0 && s - 1 + utf8_len((unsigned char)out[s - 1]) > end)
                end = s - 1;
        }
        out[end] = 0;
    }
    return w.len;
}

const char *pk_piece(const pk_model *m, int32_t id, size_t *len) {
    if (id < 0 || (uint32_t)id >= m->n_pieces) {
        *len = 0;
        return "";
    }
    *len = m->piece_off[id + 1] - m->piece_off[id];
    return m->piece_text + m->piece_off[id];
}

size_t pk_words(const pk_model *m, const pk_tok *t, size_t n, pk_word *out) {
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        size_t len;
        const char *s = pk_piece(m, t[i].id, &len);
        int starts = len >= 3 && !memcmp(s, SPACE, 3);
        if (k == 0 || starts) {
            out[k].tok = (int32_t)i;
            out[k].ntok = 0;
            out[k].start = t[i].frame;
            k++;
        }
        out[k - 1].ntok++;
        out[k - 1].end = t[i].frame + t[i].dur;
    }
    return k;
}
