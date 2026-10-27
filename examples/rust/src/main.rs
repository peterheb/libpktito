//! The library from Rust, through its own `extern "C"`: examples/transcribe.c
//! line for line, a WAV or AIFF file to its transcript, cut at its silences
//! when it is long.
//!
//!     make libpktito.a
//!     cd examples/rust && cargo run --release -- ../../models/parakeet-tdt-0.6b-v2.pkf talk.wav
//!
//! prints each chunk as "start end text", the times in seconds. A 16 kHz file
//! only, as there. build.rs links libpktito.a, the archive at the top of the
//! tree (or the one in PKTITO_LIB_DIR), so the program is one binary that
//! needs nothing beside it but the model. No crate: the declarations below
//! are core/pktito.h's and core/wav.h's, written out by hand, and nothing
//! checks them against the headers - a type written wrong here is undefined
//! behaviour there. A binding to keep would have bindgen write them.
//!
//! An example of the wiring, not a crate. `make check-ffi` holds its
//! transcript to ./pktito-c's.

use std::ffi::{c_char, c_int, c_long, CStr, CString, OsStr};
use std::process::ExitCode;

const PK_VAD_SAMPLES: usize = 512;

// The opaque handles, each a type of its own so that one cannot be passed
// for another: Rust's idiom for a C struct whose fields are the library's.
#[repr(C)]
struct PkModel([u8; 0]);
#[repr(C)]
struct PkAsr([u8; 0]);
#[repr(C)]
struct PkVad([u8; 0]);
#[repr(C)]
struct PkWav([u8; 0]);

// The structs the calls pass, field for field: #[repr(C)] lays them out as
// the C compiler does.
#[repr(C)]
#[derive(Default)]
struct PkWavInfo {
    rate: u32,
    channels: u32,
    format: u32,
    bytes: u32,
    frames: u64,
}

#[repr(C)]
#[derive(Default)]
struct PkSegParams {
    threshold: f64,
    neg_threshold: f64,
    min_speech_ms: f64,
    max_speech_s: f64,
    min_silence_ms: f64,
    speech_pad_ms: f64,
    min_silence_at_max_speech_ms: f64,
    use_max_poss_sil_at_max_speech: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PkSeg {
    start: i64,
    end: i64,
}

#[repr(C)]
struct PkTok {
    id: i32,
    frame: i32,
    dur: i32,
}

extern "C" {
    fn pk_model_load(path: *const c_char, err: *mut c_char, errlen: usize) -> *mut PkModel;
    fn pk_model_rate(m: *const PkModel) -> c_int;
    fn pk_model_free(m: *mut PkModel);
    fn pk_wav_open(path: *const c_char, info: *mut PkWavInfo, err: *mut c_char, errlen: usize) -> *mut PkWav;
    fn pk_wav_read(w: *mut PkWav, out: *mut f32, n: usize) -> usize;
    fn pk_wav_close(w: *mut PkWav);
    fn pk_vad_new(m: *const PkModel) -> *mut PkVad;
    fn pk_vad_probs(v: *mut PkVad, x: *const f32, n: usize, probs: *mut f32);
    fn pk_vad_free(v: *mut PkVad);
    fn pk_seg_defaults(p: *mut PkSegParams);
    fn pk_segments(probs: *const f32, n: usize, rate: c_int, n_samples: i64, p: *const PkSegParams,
                   out: *mut PkSeg, cap: usize) -> usize;
    fn pk_chunks(x: *const f32, speech: *const PkSeg, n_speech: usize, total: i64, max_len: i64,
                 out: *mut PkSeg) -> usize;
    fn pk_asr_new(m: *const PkModel, threads: c_int) -> *mut PkAsr;
    fn pk_asr_run(a: *mut PkAsr, x: *const f32, n: usize, toks: *mut *const PkTok) -> c_long;
    fn pk_asr_free(a: *mut PkAsr);
    fn pk_text(m: *const PkModel, t: *const PkTok, n: usize, out: *mut c_char, cap: usize) -> usize;
}

/// A path as the library takes one: UTF-8 on every system, which on Unix is
/// the name's bytes as they are.
fn c_path(p: &OsStr) -> CString {
    #[cfg(unix)]
    let bytes = std::os::unix::ffi::OsStrExt::as_bytes(p).to_vec();
    #[cfg(not(unix))]
    let bytes = p.to_str().expect("a path in Unicode").as_bytes().to_vec();
    CString::new(bytes).expect("a path without a NUL")
}

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        eprintln!("usage: transcribe MODEL.pkf FILE.wav");
        return ExitCode::from(2);
    }
    // Every call into the library is unsafe to Rust, which cannot see what
    // C does with what it is given; the safety is the header's contract.
    unsafe {
        let mut err = [0 as c_char; 256];
        let m = pk_model_load(c_path(&args[1]).as_ptr(), err.as_mut_ptr(), err.len());
        if m.is_null() {
            eprintln!("{}", CStr::from_ptr(err.as_ptr()).to_string_lossy());
            return ExitCode::from(1);
        }
        let rate = pk_model_rate(m) as usize;

        // the whole recording, as float in [-1, 1]
        let mut info = PkWavInfo::default();
        let w = pk_wav_open(c_path(&args[2]).as_ptr(), &mut info, err.as_mut_ptr(), err.len());
        if w.is_null() || info.rate as usize != rate {
            if w.is_null() {
                eprintln!("{}", CStr::from_ptr(err.as_ptr()).to_string_lossy());
            } else {
                eprintln!("want a 16 kHz file");
            }
            return ExitCode::from(1);
        }
        let mut x: Vec<f32> = Vec::new();
        let mut block = vec![0f32; 1 << 20];
        loop {
            let k = pk_wav_read(w, block.as_mut_ptr(), block.len());
            if k == 0 {
                break;
            }
            x.extend_from_slice(&block[..k]);
        }
        pk_wav_close(w);
        let n = x.len();

        // Silero's speech segments, then chunks of at most 30 s cut between
        // them, the silence left out; a recording that fits is one chunk
        let frames = n.div_ceil(PK_VAD_SAMPLES);
        let mut padded = x.clone();
        padded.resize(frames * PK_VAD_SAMPLES, 0.0);
        let mut probs = vec![0f32; frames];
        let mut seg = vec![PkSeg::default(); 2 * frames + 1];
        let v = pk_vad_new(m);
        pk_vad_probs(v, padded.as_ptr(), frames, probs.as_mut_ptr());
        let mut sp = PkSegParams::default();
        pk_seg_defaults(&mut sp);
        sp.max_speech_s = 30.0 - 0.06 - 0.1; // a segment and Silero's 30 ms pads fit in 30 s, as ./pktito-c has it
        let ns = pk_segments(probs.as_ptr(), frames, rate as c_int, n as i64, &sp, seg.as_mut_ptr(), seg.len());
        let mut chunk = vec![PkSeg::default(); 2 * ns + 2 + n / (30 * rate)];
        let nc = pk_chunks(x.as_ptr(), seg.as_ptr(), ns, n as i64, (30 * rate) as i64, chunk.as_mut_ptr());

        // each chunk through the transcriber, as NeMo would hear it as a file
        let a = pk_asr_new(m, 0);
        let mut t: *const PkTok = std::ptr::null();
        let mut text = vec![0 as c_char; 65536];
        for c in &chunk[..nc] {
            let xs = &x[c.start as usize..c.end as usize];
            let nt = pk_asr_run(a, xs.as_ptr(), xs.len(), &mut t);
            if nt < 0 {
                return ExitCode::from(1);
            }
            pk_text(m, t, nt as usize, text.as_mut_ptr(), text.len());
            println!("{:.2} {:.2} {}", c.start as f64 / rate as f64, c.end as f64 / rate as f64,
                     CStr::from_ptr(text.as_ptr()).to_string_lossy());
        }
        pk_asr_free(a);
        pk_vad_free(v);
        pk_model_free(m);
    }
    ExitCode::SUCCESS
}
