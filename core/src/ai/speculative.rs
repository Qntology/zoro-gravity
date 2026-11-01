//! 자기 추측 디코딩 (Self-Speculative Decoding): 같은 가중치로 초안을 쓰고 자기회귀로 검증합니다. gocr 와 zoro 가 함께 씁니다.
//!
//! 블록 하나의 흐름:
//! 1. **초안**: `[다음 토큰, 마스크 × (n−1)]` 을 블록 주의로 한 번에 계산합니다. 출력 i 는 위치 i+1 의 토큰입니다.
//! 2. **조기 종료**: 초안이 끝 토큰까지 닿았고 끝 토큰을 포함한 모든 토큰이 기준 확률 이상이면 검증 없이 확정합니다.
//!    짧은 조각(계좌번호 등)은 초안 한 번으로 끝납니다. 끝에 닿지 않은 블록은 어차피 실제 토큰의 키 · 값을 캐시에 넣는 계산이
//!    필요해서, 그 계산이 곧 검증입니다.
//! 3. **검증**: 초안 토큰을 인과 주의로 한 번에 읽어, 왼쪽부터 자기회귀 예측과 일치하는 만큼만 채택합니다.
//!    처음 어긋난 자리는 자기회귀 예측이 다음 블록의 첫 토큰이 됩니다.
//!
//! 검증을 거친 토큰은 한 토큰씩 자기회귀로 읽은 결과와 같습니다. 초안이 틀려도 결과가 틀리지 않고 속도만 달라집니다.
//! 어긋날 수 있는 것은 조기 종료로 확정한 마지막 블록뿐입니다. 모델 가중치는 모드마다 따로 있지 않아서 추가로 올릴 것이 없습니다.
//!
//! [`decode`] 는 초안과 검증이 각각 순전파 한 번입니다 (gocr). [`decode_stream`] 은 둘을 순전파 하나로 합칩니다 (zoro).
//! 인과 주의에서는 행의 출력이 그 행까지의 입력에만 달려 있으므로, 검증할 토큰 뒤에 마스크를 이어 붙여도 앞쪽 행은 자기회귀 값 그대로이고,
//! 마스크 행의 출력이 다음 후보가 됩니다. 후보는 문맥에서 같은 토큰 열을 찾아서도 얻습니다 (요약은 원문의 표현을 옮겨 적는 일이 많습니다).

use std::path::Path;

use anyhow::{bail, Context, Result};

use super::weights::WeightFile;

/// 초안 블록 크기를 담는 GGUF 메타데이터
pub const META_BLOCK_SIZE: &str = "zorogravity.block_diffusion.size";
/// 마스크 토큰 번호를 담는 GGUF 메타데이터
pub const META_MASK_ID: &str = "zorogravity.block_diffusion.mask_id";

/// 블록 디퓨전 설정. 모델을 학습할 때의 값이라 GGUF 에 넣어 둡니다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockDiffusion {
    pub bd_size: usize,
    pub mask_id: u32,
}

impl BlockDiffusion {
    /// GGUF 메타데이터의 값, 없으면 `model_dir/block_diffusion.json` 의 값입니다. 둘 다 없으면 초안 단계를 쓰지 않습니다.
    pub fn resolve(file: &WeightFile, model_dir: &Path) -> Result<Option<Self>> {
        match Self::from_file(file)? {
            Some(found) => Ok(Some(found)),
            None => Self::from_dir(model_dir),
        }
    }

    pub fn from_file(file: &WeightFile) -> Result<Option<Self>> {
        if !file.has_meta(META_BLOCK_SIZE) {
            return Ok(None);
        }
        Ok(Some(Self { bd_size: file.meta_usize(META_BLOCK_SIZE)?, mask_id: file.meta_usize(META_MASK_ID)? as u32 }))
    }

    pub fn from_dir(model_dir: &Path) -> Result<Option<Self>> {
        let path = model_dir.join("block_diffusion.json");
        if !path.exists() {
            return Ok(None);
        }
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)
            .with_context(|| format!("{} 를 읽을 수 없습니다", path.display()))?;
        let int = |key: &str| json[key].as_u64().with_context(|| format!("{} 에 {key} 가 없습니다", path.display()));
        Ok(Some(Self { bd_size: int("bd_size")? as usize, mask_id: int("mask_id")? as u32 }))
    }

    /// 어휘 크기에 맞는 값인지 확인합니다. 마스크 번호가 어휘 밖이면 초안을 쓸 수 없습니다.
    pub fn validate(self, vocab: usize) -> Result<Self> {
        if self.bd_size < 2 {
            bail!("블록 크기 {} 로는 초안을 쓸 수 없습니다", self.bd_size);
        }
        if self.mask_id as usize >= vocab {
            bail!("마스크 토큰 {} 이 어휘 크기 {vocab} 를 넘습니다", self.mask_id);
        }
        Ok(self)
    }
}

/// 문맥에서 후보를 찾을 때 맞춰 보는 토큰 수
const LOOKUP_NGRAM: usize = 2;

/// 순전파 하나에 실어 보내는 추측의 양. 모두 0 이면 한 토큰씩 읽습니다.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Speculation {
    /// 검증할 토큰 뒤에 이어 읽는 마스크 수. 이 행의 출력이 다음 순전파의 후보입니다.
    pub masks: usize,
    /// 문맥에서 찾아 올 후보 수의 상한
    pub lookup: usize,
    /// 순전파 하나의 행 수 상한 (검증할 토큰 + 후보 + 마스크). 0 이면 제한하지 않습니다.
    pub width: usize,
}

impl Speculation {
    pub const NONE: Self = Self { masks: 0, lookup: 0, width: 0 };

    pub fn is_none(self) -> bool {
        self == Self::NONE
    }
}

/// `seen` 의 끝 토큰들이 앞에서 마지막으로 나온 자리를 찾아, 그 뒤 토큰을 `limit` 개까지 돌려줍니다.
fn lookup(seen: &[u32], limit: usize) -> Vec<u32> {
    if limit == 0 || seen.len() <= LOOKUP_NGRAM {
        return Vec::new();
    }
    let end = seen.len() - LOOKUP_NGRAM;
    let key = &seen[end..];
    match (0..end).rev().find(|&s| seen[s..s + LOOKUP_NGRAM] == *key) {
        Some(s) => {
            let from = s + LOOKUP_NGRAM;
            seen[from..(from + limit).min(seen.len())].to_vec()
        }
        None => Vec::new(),
    }
}

/// [`decode_stream`] 이 모델에 바라는 두 가지
pub trait StreamModel {
    /// `tokens` 를 왼쪽부터 읽고 마스크 `masks` 개를 이어 읽어, 행마다 다음 토큰의 (토큰, 확률) 을 돌려줍니다.
    /// 각 행의 출력은 그 행까지의 입력에만 달려 있어야 합니다 (인과 주의).
    fn read(&mut self, tokens: &[u32], masks: usize) -> Result<Vec<(u32, f32)>>;
    /// 방금 읽은 토큰 앞 `keep` 개를 문맥에 넣습니다.
    fn commit(&mut self, keep: usize) -> Result<()>;
}

#[derive(Debug, Clone, Copy)]
pub struct StreamOptions<'a> {
    pub eos: &'a [u32],
    pub max_tokens: usize,
    pub speculation: Speculation,
    /// 후보를 찾을 문맥 (프롬프트)
    pub context: &'a [u32],
    /// 지금까지 쓴 토큰을 보고 그만 쓸지 정합니다. 멈추는 자리는 추측의 양과 무관합니다.
    pub halt: Option<fn(&[u32]) -> bool>,
}

/// 검증과 초안을 순전파 하나로 합친 디코딩. 모든 토큰을 자기회귀로 검증하므로 추측의 양과 상관없이 글이 같습니다.
/// `first` 는 프리필의 마지막 출력에서 고른 첫 토큰과 그 확률입니다.
pub fn decode_stream(model: &mut impl StreamModel, first: (u32, f32), opts: &StreamOptions<'_>) -> Result<Decoded> {
    let mut out = Decoded { truncated: true, ..Decoded::default() };
    let (mut next, mut next_conf) = first;
    let mut seen = opts.context.to_vec();
    let mut cand: Vec<u32> = Vec::new();
    while out.tokens.len() < opts.max_tokens {
        if opts.eos.contains(&next) {
            out.truncated = false;
            break;
        }
        seen.push(next);
        if cand.is_empty() {
            cand = lookup(&seen, opts.speculation.lookup);
        }
        let room = opts.max_tokens - out.tokens.len() - 1;
        let width = if opts.speculation.width == 0 { usize::MAX } else { opts.speculation.width };
        cand.truncate(room.min(width - 1));
        if let Some(end) = cand.iter().position(|t| opts.eos.contains(t)) {
            cand.truncate(end);
        }
        // 후보를 모두 채택하면 그다음 토큰까지 정해지므로, 그 뒤 자리가 남을 때만 마스크를 붙입니다.
        let masks = opts.speculation.masks.min(room.saturating_sub(cand.len() + 1)).min(width - 1 - cand.len());
        let mut tokens = Vec::with_capacity(1 + cand.len());
        tokens.push(next);
        tokens.extend_from_slice(&cand);
        let rows = model.read(&tokens, masks)?;
        out.verifies += 1;
        let mut m = 0;
        while m < cand.len() && rows[m].0 == cand[m] {
            m += 1;
        }
        out.drafted += cand.len();
        out.accepted += m;
        for i in 0..=m {
            let (t, c) = if i == 0 { (next, next_conf) } else { (cand[i - 1], rows[i - 1].1) };
            out.tokens.push(t);
            out.confs.push(c);
            if i > 0 {
                seen.push(t);
            }
            if opts.halt.is_some_and(|halt| halt(&out.tokens)) {
                out.halted = true;
                return Ok(out);
            }
        }
        model.commit(m + 1)?;
        (next, next_conf) = rows[m];
        // 후보가 모두 맞았으면 마스크 행은 맞는 문맥 위에서 계산한 값입니다. 첫 마스크는 `next` 자리이므로 출력은 그 다음 토큰입니다.
        let rest: Vec<u32> = if m == cand.len() { rows[cand.len() + 1..].iter().map(|r| r.0).collect() } else { Vec::new() };
        cand = rest;
    }
    if out.tokens.len() >= opts.max_tokens && opts.eos.contains(&next) {
        out.truncated = false;
    }
    Ok(out)
}

/// 디코딩 루프가 모델에 바라는 세 가지
pub trait BlockModel {
    /// `first` 와 마스크 `n−1` 개를 읽어 위치 1..=n 의 (토큰, 확률) 을 돌려줍니다. 문맥은 바뀌지 않습니다.
    fn draft(&mut self, first: u32, n: usize) -> Result<Vec<(u32, f32)>>;
    /// `candidates` 를 왼쪽부터 읽어 각 토큰 다음 토큰의 (토큰, 확률) 을 돌려줍니다.
    fn verify(&mut self, candidates: &[u32]) -> Result<Vec<(u32, f32)>>;
    /// 방금 검증한 토큰 앞 `keep` 개를 문맥에 넣습니다.
    fn commit(&mut self, keep: usize) -> Result<()>;
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeOptions<'a> {
    pub eos: &'a [u32],
    pub max_tokens: usize,
    /// 블록 크기. 1 이면 초안 없이 한 토큰씩 읽습니다 (순수 자기회귀).
    pub block: usize,
    pub early_exit: Option<f32>,
    /// 지금까지 쓴 토큰을 보고 그만 쓸지 정합니다 (같은 말의 되풀이 등). 토큰 하나를 쓸 때마다 부르므로 멈추는 자리는 블록 크기와 무관합니다.
    pub halt: Option<fn(&[u32]) -> bool>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Decoded {
    pub tokens: Vec<u32>,
    pub confs: Vec<f32>,
    pub truncated: bool,
    pub drafts: usize,
    pub verifies: usize,
    pub early_exit: bool,
    /// `halt` 가 멈췄는가
    pub halted: bool,
    /// 검증에 넘긴 초안 토큰 수와 그중 채택한 수
    pub drafted: usize,
    pub accepted: usize,
}

/// `first` 는 프리필의 마지막 출력에서 고른 첫 토큰과 그 확률입니다.
pub fn decode(model: &mut impl BlockModel, first: (u32, f32), opts: &DecodeOptions<'_>) -> Result<Decoded> {
    let mut out = Decoded { truncated: true, ..Decoded::default() };
    let (mut next, mut next_conf) = first;
    while out.tokens.len() < opts.max_tokens {
        if opts.eos.contains(&next) {
            out.truncated = false;
            break;
        }
        let n = opts.block.max(1).min(opts.max_tokens - out.tokens.len());
        // 후보: next 와 초안이 내놓은 토큰들 (끝 토큰 앞까지)
        let mut cand = vec![next];
        let mut cand_conf = vec![next_conf];
        let mut reached_end = false;
        if n > 1 {
            let draft = model.draft(next, n)?;
            out.drafts += 1;
            let mut end_conf = 1.0f32;
            for &(t, c) in &draft[..n - 1] {
                if opts.eos.contains(&t) {
                    reached_end = true;
                    end_conf = c;
                    break;
                }
                cand.push(t);
                cand_conf.push(c);
            }
            if !reached_end {
                // 블록의 마지막 출력은 블록 다음 자리의 토큰입니다.
                let (t, c) = draft[n - 1];
                if opts.eos.contains(&t) {
                    reached_end = true;
                    end_conf = c;
                }
            }
            if let Some(tau) = opts.early_exit {
                if reached_end && end_conf >= tau && cand_conf[1..].iter().all(|&c| c >= tau) {
                    out.tokens.extend_from_slice(&cand);
                    out.confs.extend_from_slice(&cand_conf);
                    out.truncated = false;
                    out.early_exit = true;
                    return Ok(out);
                }
            }
        }
        let ar = model.verify(&cand)?;
        out.verifies += 1;
        // 왼쪽부터 자기회귀 예측과 일치하는 만큼 채택합니다.
        let mut m = 0;
        while m + 1 < cand.len() && ar[m].0 == cand[m + 1] {
            m += 1;
        }
        out.drafted += cand.len() - 1;
        out.accepted += m;
        for i in 0..=m {
            out.tokens.push(cand[i]);
            out.confs.push(if i == 0 { cand_conf[0] } else { ar[i - 1].1 });
            if opts.halt.is_some_and(|halt| halt(&out.tokens)) {
                out.halted = true;
                return Ok(out);
            }
        }
        model.commit(m + 1)?;
        (next, next_conf) = ar[m];
    }
    if out.tokens.len() >= opts.max_tokens && opts.eos.contains(&next) {
        out.truncated = false;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: u32 = 999;

    /// 정답 수열을 아는 대역. 초안은 `wrong` 위치에서 틀리고, 검증(자기회귀)은 항상 정답을 냅니다.
    struct Oracle {
        truth: Vec<u32>,
        at: usize,
        wrong: Vec<usize>,
        draft_conf: f32,
        drafts: Vec<(u32, usize)>,
        verified: Vec<Vec<u32>>,
        committed: usize,
    }

    impl Oracle {
        fn new(truth: &[u32], wrong: &[usize], draft_conf: f32) -> Self {
            let mut t = truth.to_vec();
            t.push(EOS);
            Self { truth: t, at: 0, wrong: wrong.to_vec(), draft_conf, drafts: vec![], verified: vec![], committed: 0 }
        }

        fn truth_at(&self, i: usize) -> u32 {
            self.truth.get(i).copied().unwrap_or(EOS)
        }
    }

    impl BlockModel for Oracle {
        fn draft(&mut self, first: u32, n: usize) -> Result<Vec<(u32, f32)>> {
            self.drafts.push((first, n));
            Ok((1..=n)
                .map(|i| {
                    let p = self.at + i;
                    if self.wrong.contains(&p) { (500 + p as u32, self.draft_conf) } else { (self.truth_at(p), self.draft_conf) }
                })
                .collect())
        }

        fn verify(&mut self, candidates: &[u32]) -> Result<Vec<(u32, f32)>> {
            self.verified.push(candidates.to_vec());
            Ok((1..=candidates.len()).map(|i| (self.truth_at(self.at + i), 0.5)).collect())
        }

        fn commit(&mut self, keep: usize) -> Result<()> {
            self.at += keep;
            self.committed += keep;
            Ok(())
        }
    }

    fn opts(block: usize, early: Option<f32>) -> DecodeOptions<'static> {
        DecodeOptions { eos: &[EOS], max_tokens: 100, block, early_exit: early, halt: None }
    }

    fn run(truth: &[u32], wrong: &[usize], conf: f32, block: usize, early: Option<f32>) -> (Decoded, Oracle) {
        let mut o = Oracle::new(truth, wrong, conf);
        let first = o.truth[0];
        let out = decode(&mut o, (first, 0.9), &opts(block, early)).unwrap();
        (out, o)
    }

    #[test]
    fn a_correct_draft_needs_one_verification_per_block() {
        let truth: Vec<u32> = (1..=10).collect();
        let (out, o) = run(&truth, &[], 0.5, 4, None);
        assert_eq!(out.tokens, truth);
        assert!(!out.truncated);
        assert_eq!(out.drafts, 3, "토큰 10개를 블록 4 로: {:?}", o.drafts);
        assert_eq!(out.verifies, 3);
    }

    #[test]
    fn a_wrong_draft_keeps_only_the_matching_prefix_and_the_result_stays_correct() {
        let truth: Vec<u32> = (1..=12).collect();
        let (out, o) = run(&truth, &[2, 7], 0.5, 4, None);
        assert_eq!(out.tokens, truth, "초안이 틀려도 결과는 자기회귀와 같습니다");
        assert_eq!(out.confs.len(), truth.len());
        // 첫 블록: [1, 2, wrong, ..] → 위치 0,1 만 채택
        assert_eq!(o.verified[0][..2], [1, 2]);
        assert_ne!(o.verified[0][2], 3);
        assert_eq!(o.committed, 12);
        assert!(out.accepted < out.drafted, "틀린 초안은 채택 수에 들어가지 않습니다");
    }

    #[test]
    fn block_size_one_is_pure_autoregression() {
        let truth: Vec<u32> = (1..=5).collect();
        let (out, o) = run(&truth, &[], 0.5, 1, None);
        assert_eq!(out.tokens, truth);
        assert_eq!((out.drafts, out.verifies), (0, 5));
        assert_eq!((out.drafted, out.accepted), (0, 0));
        assert!(o.drafts.is_empty());
    }

    #[test]
    fn a_confident_final_draft_exits_without_verification() {
        let truth: Vec<u32> = (1..=3).collect();
        let (out, o) = run(&truth, &[], 0.995, 8, Some(0.99));
        assert_eq!(out.tokens, truth);
        assert!(out.early_exit);
        assert_eq!((out.drafts, out.verifies), (1, 0), "검증 없이 끝납니다");
        assert!(o.verified.is_empty());
    }

    #[test]
    fn an_unsure_final_draft_is_verified() {
        let truth: Vec<u32> = (1..=3).collect();
        let (out, _) = run(&truth, &[], 0.98, 8, Some(0.99));
        assert_eq!(out.tokens, truth);
        assert!(!out.early_exit);
        assert_eq!(out.verifies, 1);
    }

    #[test]
    fn a_block_that_does_not_reach_the_end_is_always_verified() {
        // 확률이 높아도 끝에 닿지 않은 블록은 키 · 값을 넣는 계산이 필요하므로 검증을 거칩니다.
        let truth: Vec<u32> = (1..=20).collect();
        let (out, _) = run(&truth, &[], 0.999, 8, Some(0.99));
        assert_eq!(out.tokens, truth);
        assert!(out.verifies >= 2);
        assert!(out.early_exit, "마지막 블록만 조기 종료");
    }

    #[test]
    fn without_early_exit_every_token_is_verified_even_at_full_confidence() {
        let truth: Vec<u32> = (1..=5).collect();
        let (out, o) = run(&truth, &[], 1.0, 8, None);
        assert_eq!(out.tokens, truth);
        assert!(!out.early_exit);
        assert_eq!(o.verified.concat(), truth, "글쓰기는 모든 토큰을 검증합니다");
    }

    #[test]
    fn the_token_limit_truncates() {
        let truth: Vec<u32> = (1..=30).collect();
        let mut o = Oracle::new(&truth, &[], 0.5);
        let out = decode(&mut o, (1, 0.9), &DecodeOptions { max_tokens: 10, block: 4, ..opts(4, None) }).unwrap();
        assert_eq!(out.tokens, (1..=10).collect::<Vec<u32>>());
        assert!(out.truncated);
    }

    #[test]
    fn an_immediate_end_gives_nothing() {
        let mut o = Oracle::new(&[], &[], 0.5);
        let out = decode(&mut o, (EOS, 0.9), &opts(8, None)).unwrap();
        assert!(out.tokens.is_empty());
        assert!(!out.truncated);
        assert_eq!((out.drafts, out.verifies), (0, 0));
    }

    #[test]
    fn halting_stops_at_the_same_token_for_every_block_size() {
        fn at_seven(tokens: &[u32]) -> bool {
            tokens.last() == Some(&7)
        }
        let truth: Vec<u32> = (1..=20).collect();
        for block in [1, 2, 4, 8] {
            let mut o = Oracle::new(&truth, &[5], 0.5);
            let out = decode(&mut o, (1, 0.9), &DecodeOptions { halt: Some(at_seven), ..opts(block, None) }).unwrap();
            assert_eq!(out.tokens, (1..=7).collect::<Vec<u32>>(), "블록 {block}");
            assert_eq!(out.confs.len(), 7);
            assert!(out.halted && out.truncated, "끝 토큰으로 끝난 글이 아닙니다");
        }
    }

    /// 다음 토큰을 정답 수열에서 답하는 대역. 마스크 행은 `wrong` 위치에서 틀립니다.
    struct Stream {
        truth: Vec<u32>,
        at: usize,
        wrong: Vec<usize>,
        reads: Vec<(Vec<u32>, usize)>,
        committed: Vec<usize>,
    }

    impl Stream {
        fn new(truth: &[u32], wrong: &[usize]) -> Self {
            let mut t = truth.to_vec();
            t.push(EOS);
            Self { truth: t, at: 0, wrong: wrong.to_vec(), reads: vec![], committed: vec![] }
        }

        fn truth_at(&self, i: usize) -> u32 {
            self.truth.get(i).copied().unwrap_or(EOS)
        }
    }

    impl StreamModel for Stream {
        fn read(&mut self, tokens: &[u32], masks: usize) -> Result<Vec<(u32, f32)>> {
            self.reads.push((tokens.to_vec(), masks));
            // 어긋난 입력 뒤의 행은 진짜 모델처럼 틀린 문맥의 값이어야 하므로, 어긋난 자리까지만 정답입니다.
            let ok = (0..tokens.len()).take_while(|&i| tokens[i] == self.truth_at(self.at + i)).count();
            let mut out = Vec::new();
            for i in 0..tokens.len() + masks {
                let p = self.at + i + 1;
                let right = i < ok || (i >= tokens.len() && ok == tokens.len() && !self.wrong.contains(&p));
                out.push((if right { self.truth_at(p) } else { 700 + p as u32 }, 0.5));
            }
            Ok(out)
        }

        fn commit(&mut self, keep: usize) -> Result<()> {
            self.at += keep;
            self.committed.push(keep);
            Ok(())
        }
    }

    fn stream(truth: &[u32], wrong: &[usize], context: &[u32], masks: usize, look: usize, max_tokens: usize) -> (Decoded, Stream) {
        let mut s = Stream::new(truth, wrong);
        let first = s.truth[0];
        let opts = StreamOptions { eos: &[EOS], max_tokens, speculation: Speculation { masks, lookup: look, width: 0 }, context, halt: None };
        (decode_stream(&mut s, (first, 0.9), &opts).unwrap(), s)
    }

    #[test]
    fn a_stream_without_speculation_reads_one_token_at_a_time() {
        let truth: Vec<u32> = (1..=5).collect();
        let (out, s) = stream(&truth, &[], &[], 0, 0, 100);
        assert_eq!(out.tokens, truth);
        assert!(!out.truncated);
        assert_eq!(out.verifies, 5);
        assert!(s.reads.iter().all(|(t, m)| t.len() == 1 && *m == 0));
    }

    #[test]
    fn right_masks_save_reads_and_wrong_masks_change_nothing() {
        let truth: Vec<u32> = (1..=24).collect();
        let (right, _) = stream(&truth, &[], &[], 3, 0, 100);
        assert_eq!(right.tokens, truth);
        assert!(right.verifies * 2 < truth.len(), "{}번 읽음", right.verifies);
        assert_eq!(right.confs.len(), truth.len());

        let (wrong, s) = stream(&truth, &[3, 4, 9, 15, 16, 20], &[], 3, 0, 100);
        assert_eq!(wrong.tokens, truth, "틀린 초안은 시간만 씁니다");
        assert_eq!(s.committed.iter().sum::<usize>(), truth.len());
        assert!(wrong.accepted < wrong.drafted);
    }

    #[test]
    fn candidates_found_in_the_context_are_verified_like_any_other() {
        // 쓸 글이 문맥에 그대로 있으면 후보가 맞아서 적게 읽습니다.
        let truth: Vec<u32> = (10..30).collect();
        let (copied, _) = stream(&truth, &[], &truth, 0, 4, 100);
        assert_eq!(copied.tokens, truth);
        assert!(copied.verifies * 3 < truth.len(), "{}번 읽음", copied.verifies);
        // 문맥이 엉뚱해도 글은 같습니다.
        let (misled, _) = stream(&truth, &[], &[11, 12, 500, 501, 12, 13, 502, 503], 0, 4, 100);
        assert_eq!(misled.tokens, truth);
    }

    #[test]
    fn the_width_limits_every_read_and_keeps_the_text() {
        let truth: Vec<u32> = (10..40).collect();
        for width in [1usize, 2, 3, 4] {
            let mut s = Stream::new(&truth, &[5, 9]);
            let opts = StreamOptions { eos: &[EOS], max_tokens: 100, speculation: Speculation { masks: 2, lookup: 5, width }, context: &truth, halt: None };
            let out = decode_stream(&mut s, (10, 0.9), &opts).unwrap();
            assert_eq!(out.tokens, truth, "폭 {width}");
            assert!(s.reads.iter().all(|(t, m)| t.len() + m <= width), "폭 {width}: {:?}", s.reads);
        }
    }

    #[test]
    fn lookup_takes_the_latest_match_and_needs_the_whole_ngram() {
        assert_eq!(lookup(&[1, 2, 3, 4, 1, 2, 5, 6, 1, 2], 2), vec![5, 6], "가까운 쪽");
        assert_eq!(lookup(&[1, 2, 3, 1, 2], 9), vec![3, 1, 2]);
        assert!(lookup(&[1, 2, 3, 2, 3 + 1], 4).is_empty());
        assert!(lookup(&[1, 2], 4).is_empty());
        assert!(lookup(&[1, 2, 1, 2], 0).is_empty());
    }

    #[test]
    fn a_stream_stops_at_the_limit_and_at_the_end_token() {
        let truth: Vec<u32> = (1..=30).collect();
        for (masks, look) in [(0, 0), (1, 0), (3, 0), (0, 4), (2, 4)] {
            let (out, s) = stream(&truth, &[], &truth, masks, look, 10);
            assert_eq!(out.tokens, (1..=10).collect::<Vec<u32>>(), "마스크 {masks}, 탐색 {look}");
            assert!(out.truncated);
            assert!(s.committed.iter().sum::<usize>() <= 10);
        }
        // 후보 안의 끝 토큰은 후보에서 빠지고, 검증이 끝을 알립니다.
        let short: Vec<u32> = (1..=6).collect();
        let (out, s) = stream(&short, &[], &[1, 2, 3, 4, 5, 6, EOS, 7, 8], 0, 8, 100);
        assert_eq!(out.tokens, short);
        assert!(!out.truncated);
        assert!(s.reads.iter().all(|(t, _)| !t.contains(&EOS)));
    }

    #[test]
    fn a_stream_halts_at_the_same_token_for_every_amount_of_speculation() {
        fn at_seven(tokens: &[u32]) -> bool {
            tokens.last() == Some(&7)
        }
        let truth: Vec<u32> = (1..=20).collect();
        for (masks, look) in [(0, 0), (1, 0), (4, 0), (0, 6), (3, 6)] {
            let mut s = Stream::new(&truth, &[]);
            let opts = StreamOptions { eos: &[EOS], max_tokens: 100, speculation: Speculation { masks, lookup: look, width: 0 }, context: &truth, halt: Some(at_seven) };
            let out = decode_stream(&mut s, (1, 0.9), &opts).unwrap();
            assert_eq!(out.tokens, (1..=7).collect::<Vec<u32>>(), "마스크 {masks}, 탐색 {look}");
            assert!(out.halted && out.truncated);
        }
    }

    #[test]
    fn the_settings_are_read_from_the_file_first_and_checked_against_the_vocabulary() {
        let dir = std::env::temp_dir().join(format!("zorogravity-bd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(BlockDiffusion::from_dir(&dir).unwrap(), None);
        std::fs::write(dir.join("block_diffusion.json"), r#"{"bd_size": 8, "mask_id": 299}"#).unwrap();
        let found = BlockDiffusion::from_dir(&dir).unwrap().unwrap();
        assert_eq!(found, BlockDiffusion { bd_size: 8, mask_id: 299 });
        assert_eq!(found.validate(300).unwrap(), found);
        assert!(found.validate(299).is_err(), "마스크가 어휘 밖입니다");
        assert!(BlockDiffusion { bd_size: 1, mask_id: 1 }.validate(300).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
