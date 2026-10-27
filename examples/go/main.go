// The library from Go, through cgo: examples/transcribe.c line for line, a
// WAV or AIFF file to its transcript, cut at its silences when it is long.
//
//	make libpktito.a
//	cd examples/go && go build && ./transcribe ../../models/parakeet-tdt-0.6b-v2.pkf talk.wav
//
// prints each chunk as "start end text", the times in seconds. A 16 kHz
// file only, as there. It links libpktito.a, the archive at the top of the
// tree, so the program is one binary that needs nothing beside it but the
// model: cgo reads core/pktito.h and core/wav.h for the types, and Go's
// compiler checks each call against them.
//
// An example of the wiring, not a package. `make check-ffi` holds its
// transcript to ./pktito-c's.
package main

/*
#cgo CFLAGS: -I${SRCDIR}/../../core
#cgo LDFLAGS: ${SRCDIR}/../../libpktito.a -lm -lpthread
#include <stdlib.h>
#include "pktito.h"
#include "wav.h"
*/
import "C"

import (
	"fmt"
	"os"
	"unsafe"
)

// floats is s as C's float *: its memory, not a copy. cgo lets C have Go
// memory for the call, as long as that memory holds no Go pointers, which
// a []float32 does not.
func floats(s []float32) *C.float {
	if len(s) == 0 {
		return nil
	}
	return (*C.float)(&s[0])
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: transcribe MODEL.pkf FILE.wav")
		os.Exit(2)
	}
	os.Exit(run(os.Args[1], os.Args[2]))
}

func run(modelPath, wavPath string) int {
	// The library's paths are UTF-8 on every system, and a Go string is
	// one: os.Args has Windows' UTF-16 command line as UTF-8, whole.
	var err [256]C.char
	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))
	m := C.pk_model_load(cpath, &err[0], C.size_t(len(err)))
	if m == nil {
		fmt.Fprintln(os.Stderr, C.GoString(&err[0]))
		return 1
	}
	rate := int(C.pk_model_rate(m))

	// the whole recording, as float in [-1, 1]
	var info C.pk_wav_info
	cwav := C.CString(wavPath)
	defer C.free(unsafe.Pointer(cwav))
	w := C.pk_wav_open(cwav, &info, &err[0], C.size_t(len(err)))
	if w == nil || int(info.rate) != rate {
		if w != nil {
			fmt.Fprintln(os.Stderr, "want a 16 kHz file")
		} else {
			fmt.Fprintln(os.Stderr, C.GoString(&err[0]))
		}
		return 1
	}
	var x []float32
	block := make([]float32, 1<<20)
	for {
		k := int(C.pk_wav_read(w, floats(block), C.size_t(len(block))))
		if k == 0 {
			break
		}
		x = append(x, block[:k]...)
	}
	C.pk_wav_close(w)
	n := len(x)

	// Silero's speech segments, then chunks of at most 30 s cut between
	// them, the silence left out; a recording that fits is one chunk
	frames := (n + C.PK_VAD_SAMPLES - 1) / C.PK_VAD_SAMPLES
	padded := make([]float32, frames*C.PK_VAD_SAMPLES)
	copy(padded, x)
	probs := make([]float32, frames)
	seg := make([]C.pk_seg, 2*frames+1)
	v := C.pk_vad_new(m)
	C.pk_vad_probs(v, floats(padded), C.size_t(frames), floats(probs))
	var sp C.pk_seg_params
	C.pk_seg_defaults(&sp)
	sp.max_speech_s = 30 - 0.06 - 0.1 // a segment and Silero's 30 ms pads fit in 30 s, as ./pktito-c has it
	ns := int(C.pk_segments(floats(probs), C.size_t(frames), C.int(rate), C.int64_t(n), &sp, &seg[0], C.size_t(len(seg))))
	chunk := make([]C.pk_seg, 2*ns+2+n/(30*rate))
	nc := int(C.pk_chunks(floats(x), &seg[0], C.size_t(ns), C.int64_t(n), C.int64_t(30*rate), &chunk[0]))

	// each chunk through the transcriber, as NeMo would hear it as a file;
	// the transcriber runs its own threads, and a cgo call holds only the
	// goroutine that makes it
	a := C.pk_asr_new(m, 0)
	var t *C.pk_tok
	text := make([]C.char, 65536)
	for _, c := range chunk[:nc] {
		xs := x[c.start:c.end]
		nt := C.pk_asr_run(a, floats(xs), C.size_t(len(xs)), &t)
		if nt < 0 {
			return 1
		}
		C.pk_text(m, t, C.size_t(nt), &text[0], C.size_t(len(text)))
		fmt.Printf("%.2f %.2f %s\n", float64(c.start)/float64(rate), float64(c.end)/float64(rate), C.GoString(&text[0]))
	}
	C.pk_asr_free(a)
	C.pk_vad_free(v)
	C.pk_model_free(m)
	return 0
}
