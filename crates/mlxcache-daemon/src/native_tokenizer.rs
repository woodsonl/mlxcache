//! T13: native tokenization (R4 endgame — zero Python on the daemon hot path).
//!
//! The hot path per request is validate → **tokenize** → route → respond.
//! With the sidecar adapter, tokenize is an HTTP round trip (serialize +
//! socket + Python json + HF encode + socket + parse). When the model's
//! `tokenizer.json` is available locally, the HF `tokenizers` Rust crate
//! produces the same ids in-process — it is the same implementation that
//! backs HF's Python binding, so parity is by construction (and probed).
//!
//! Opt-in via `MLXCACHE_NATIVE_TOKENIZER=<path to tokenizer.json>`. When set,
//! the daemon answers tokenize locally and reports the file's SHA-256 as the
//! fingerprint's `tokenizer_hash`: the artifact identity R1-2 pins. A native
//! tokenize and a sidecar tokenize of DIFFERENT artifacts therefore can never
//! cross-serve — different files, different hashes.
//!
//! Fail closed: if the file is missing/unparsable at startup the daemon
//! refuses to start. A partial native rollout (some requests hashed by one
//! tokenizer, some by another, within one model) would silently corrupt
//! routing, and a silent fallback re-opens exactly that hole.

use sha2::{Digest, Sha256 as Sha2};
use std::path::Path;
use std::sync::OnceLock;

/// A native tokenizer bound to one model artifact: encode + identity.
pub struct NativeTokenizer {
    tokenizer: tokenizers::Tokenizer,
    /// sha256 hex of the tokenizer.json bytes — the R1-2 artifact identity.
    pub tokenizer_hash: String,
}

impl NativeTokenizer {
    /// Load + validate. Errors are startup-fatal (fail closed).
    pub fn load(tokenizer_json: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(tokenizer_json)
            .map_err(|e| format!("cannot read {}: {e}", tokenizer_json.display()))?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes)
            .map_err(|e| format!("cannot parse {}: {e}", tokenizer_json.display()))?;
        let mut hasher = Sha2::new();
        hasher.update(&bytes);
        let tokenizer_hash = format!("{:x}", hasher.finalize());
        Ok(Self {
            tokenizer,
            tokenizer_hash,
        })
    }

    /// Encode a prompt to token ids — identical to the sidecar's
    /// `tokenizer.encode(prompt)` for the same artifact (same library).
    pub fn encode(&self, prompt: &str) -> Vec<u32> {
        self.tokenizer
            .encode(prompt, /* add_special_tokens */ false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }
}

/// Process-wide native tokenizer, parsed once from the environment.
/// Returns `(path, tokenizer)` when `MLXCACHE_NATIVE_TOKENIZER` is set and
/// usable; panics at first use when set and unusable (fail closed).
pub fn from_env() -> Option<(String, &'static NativeTokenizer)> {
    static NATIVE: OnceLock<Option<(String, Box<NativeTokenizer>)>> = OnceLock::new();
    let slot = NATIVE.get_or_init(|| {
        let path = std::env::var("MLXCACHE_NATIVE_TOKENIZER").ok()?;
        match NativeTokenizer::load(Path::new(&path)) {
            Ok(t) => {
                tracing::info!(
                    path = %path,
                    hash = %t.tokenizer_hash,
                    "native tokenizer loaded (T13): tokenize bypasses the sidecar"
                );
                Some((path, Box::new(t)))
            }
            Err(e) => panic!(
                "MLXCACHE_NATIVE_TOKENIZER={path} unusable: {e} — fix the path or unset it \
                 (a mixed tokenize path would corrupt routing, so the daemon refuses to start)"
            ),
        }
    });
    slot.as_ref().map(|(path, t)| (path.clone(), t.as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tokenizer_json(dir: &Path, contents: &str) -> std::path::PathBuf {
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, contents).unwrap();
        path
    }

    // A minimal-but-real tokenizer.json: a Whitespace pre-tokenizer + WordLevel
    // model over a tiny vocab. Deterministic ids, no downloads.
    const WORDLEVEL_JSON: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": { "hello": 0, "world": 1, "the": 2, "fox": 3, "[UNK]": 4 },
        "unk_token": "[UNK]"
      }
    }"#;

    #[test]
    fn loads_and_hashes_the_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_tokenizer_json(dir.path(), WORDLEVEL_JSON);
        let nt = NativeTokenizer::load(&path).unwrap();
        assert_eq!(nt.tokenizer_hash.len(), 64);
        // Deterministic: same bytes → same hash.
        let nt2 = NativeTokenizer::load(&path).unwrap();
        assert_eq!(nt.tokenizer_hash, nt2.tokenizer_hash);
        // Different bytes → different hash (R1-2 artifact identity).
        let other = dir.path().join("other-tokenizer.json");
        std::fs::write(&other, WORDLEVEL_JSON.replace("world\": 1", "world\": 9")).unwrap();
        let nt3 = NativeTokenizer::load(&other).unwrap();
        assert_ne!(nt.tokenizer_hash, nt3.tokenizer_hash);
    }

    #[test]
    fn encode_is_stable_and_unk_aware() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_tokenizer_json(dir.path(), WORDLEVEL_JSON);
        let nt = NativeTokenizer::load(&path).unwrap();
        let ids = nt.encode("hello world fox");
        assert_eq!(ids, vec![0, 1, 3]);
        // Unknown words must not panic; WordLevel maps the whole pre-token
        // to [UNK] (the id after the 4 vocab entries).
        let ids = nt.encode("zebra");
        assert_eq!(ids, vec![4]);
    }

    #[test]
    fn empty_prompt_encodes_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_tokenizer_json(dir.path(), WORDLEVEL_JSON);
        let nt = NativeTokenizer::load(&path).unwrap();
        assert!(nt.encode("").is_empty());
    }

    #[test]
    fn unreadable_file_is_a_load_error_not_a_panic() {
        let err = NativeTokenizer::load(Path::new("/nonexistent/tokenizer.json"))
            .err()
            .expect("must fail");
        assert!(err.contains("cannot read"), "got: {err}");
    }

    #[test]
    fn unparsable_file_is_a_load_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, b"not json at all").unwrap();
        let err = NativeTokenizer::load(&path).err().expect("must fail");
        assert!(err.contains("cannot parse"), "got: {err}");
    }
}
