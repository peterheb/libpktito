/* pathcheck: the library's paths are UTF-8 on every system (core/os.h).
 *
 * In a directory named past ASCII and past Latin-1, tests/pathcheck-tést-ñ-я,
 * which this test makes itself:
 *
 *   wav       pk_wav_create writes ñ.wav and pk_wav_open reads it back, the
 *             same samples; and the file is there under that name, asked
 *             for by the system's own call (Windows: _wfopen, in UTF-16)
 *   model     tiny.pkf, copied in as tiny-é.pkf by the system's own call,
 *             loads through pk_model_load
 *   not UTF-8 a path whose bytes are not UTF-8 is refused with a reason,
 *             and on Windows the reason says so
 *
 * On POSIX a path is its bytes and this holds by construction. On Windows
 * it is the regression test: the ANSI functions read UTF-8 in the code
 * page, look for tÃ©st-Ã±-Ñ? and find nothing. A round trip through the
 * library alone would pass even so, which is why the directory and the
 * copy are made, and the WAV looked for, without it.
 *
 *     ./tests/pathcheck tests/data/tiny.pkf
 */
#include "pktito.h"
#include "wav.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <wchar.h>
#define DIR_N  L"tests\\pathcheck-t\u00e9st-\u00f1-\u044f"
#define WAV_N  DIR_N L"\\\u00f1.wav"
#define PKF_N  DIR_N L"\\tiny-\u00e9.pkf"
#define N(s)   L##s
#define native_fopen _wfopen
#define native_mkdir(d) _wmkdir(d)
#define native_rmdir(d) _wrmdir(d)
#define native_remove(f) _wremove(f)
#else
#include <sys/stat.h>
#include <unistd.h>
#define DIR_N  "tests/pathcheck-t\xc3\xa9st-\xc3\xb1-\xd1\x8f"
#define WAV_N  DIR_N "/\xc3\xb1.wav"
#define PKF_N  DIR_N "/tiny-\xc3\xa9.pkf"
#define N(s)   s
#define native_fopen fopen
#define native_mkdir(d) mkdir(d, 0777)
#define native_rmdir(d) rmdir(d)
#define native_remove(f) remove(f)
#endif

/* the same names in UTF-8, as the library takes them */
#define DIR8 "tests/pathcheck-t\xc3\xa9st-\xc3\xb1-\xd1\x8f"
#define WAV8 DIR8 "/\xc3\xb1.wav"
#define PKF8 DIR8 "/tiny-\xc3\xa9.pkf"

static int fails, cases;

#define FAIL(...) do { fails++; printf("FAIL "); printf(__VA_ARGS__); printf("\n"); } while (0)

/* src to dst by the system's own call, so that the name on disk is the
 * one the test means whatever the library does */
static int copy_in(const char *src) {
    FILE *in = fopen(src, "rb"), *out = native_fopen(PKF_N, N("wb"));
    char b[65536];
    size_t k;
    int ok = in && out;
    while (ok && (k = fread(b, 1, sizeof b, in)) > 0)
        ok = fwrite(b, 1, k, out) == k;
    if (in)
        fclose(in);
    if (out && fclose(out))
        ok = 0;
    return ok;
}

int main(int argc, char **argv) {
    const char *tiny = argc > 1 ? argv[1] : "tests/data/tiny.pkf";
    char err[512];

    if (native_mkdir(DIR_N) != 0 && errno != EEXIST) {
        printf("pathcheck: cannot make %s: %s\n", DIR8, strerror(errno));
        return 2;
    }

    /* ---- wav: written and read back by the library, found by the system */
    {
        float x[480], y[480];
        for (int i = 0; i < 480; i++)
            x[i] = (float)((i * 37) % 200 - 100) / 128.0f;   /* exact in 16 bits */
        cases++;
        pk_wav_out *o = pk_wav_create(WAV8, 16000, 0, err, sizeof err);
        if (!o || pk_wav_write(o, x, 480) || pk_wav_finish(o, err, sizeof err)) {
            FAIL("pk_wav_create %s: %s", WAV8, err);
        } else {
            pk_wav_info info;
            pk_wav *w = pk_wav_open(WAV8, &info, err, sizeof err);
            size_t n = w ? pk_wav_read(w, y, 480) : 0;
            if (!w)
                FAIL("pk_wav_open %s: %s", WAV8, err);
            else if (n != 480 || memcmp(x, y, sizeof x))
                FAIL("%s: read back %zu frames, not the 480 written", WAV8, n);
            pk_wav_close(w);
            FILE *f = native_fopen(WAV_N, N("rb"));
            cases++;
            if (!f)
                FAIL("%s: written, but not there under that name", WAV8);
            else
                fclose(f);
        }
    }

    /* ---- model: copied in by the system, loaded by the library */
    cases++;
    if (!copy_in(tiny)) {
        FAIL("cannot copy %s to %s", tiny, PKF8);
    } else {
        pk_model *m = pk_model_load(PKF8, err, sizeof err);
        if (!m)
            FAIL("pk_model_load %s: %s", PKF8, err);
        pk_model_free(m);                   /* before the remove: Windows keeps a mapped file */
    }

    /* ---- not UTF-8: a lone continuation byte, and a lead byte with none */
    {
        static const char *bad[] = { "tests/\x80.wav", "tests/t\xc3st.wav" };
        for (int i = 0; i < 2; i++) {
            pk_wav_info info;
            err[0] = 0;
            pk_wav *w = pk_wav_open(bad[i], &info, err, sizeof err);
            cases++;
            if (w) {
                FAIL("pk_wav_open on bytes that are not UTF-8: opened");
                pk_wav_close(w);
            } else if (!err[0]) {
                FAIL("pk_wav_open on bytes that are not UTF-8: no reason");
            }
#ifdef _WIN32
            else if (!strstr(err, "UTF-8"))
                FAIL("pk_wav_open on bytes that are not UTF-8: \"%s\"", err);
            err[0] = 0;
            pk_model *m = pk_model_load(bad[i], err, sizeof err);
            cases++;
            if (m || !strstr(err, "UTF-8"))
                FAIL("pk_model_load on bytes that are not UTF-8: %s", m ? "loaded" : err);
            pk_model_free(m);
#endif
        }
    }

    native_remove(WAV_N);
    native_remove(PKF_N);
    native_rmdir(DIR_N);
    if (fails) {
        printf("pathcheck: %d of %d cases wrong\n", fails, cases);
        return 1;
    }
    printf("pathcheck: %d cases, a WAV and a model under %s\n", cases, DIR8);
    return 0;
}
