//! jev_head 분석: 가중치 구조, 크기, 점수 규모를 살펴봅니다.
//!
//! 실행: cargo run -p zorogravity-core --example jev_head_probe -- models/zoro
//!
//! 알고 싶은 것
//! 1. 가중치가 학습된 구조인가 (무작위 초기값 그대로가 아닌가)
//! 2. 점수 s = q·k 를 sqrt(256) 으로 나누는 것이 그럴듯한가
//!
//! 백본 없이 가중치만 보는 점검이라서, 최종 정규화 가중치(model.language_model.norm.weight)로 은닉 상태의 규모를 추정합니다.
//! 최종 RMSNorm 뒤의 은닉 상태는 `h_i = w_i * z_i` 이고 z 의 제곱평균은 1 이므로, h = w ⊙ z (z ~ N(0,1)) 로 규모만 흉내 냅니다.
//! 이것은 점수의 크기를 어림하는 것이지, 학습된 분포를 재현하는 것이 아닙니다.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax_last_dim;
use zorogravity_core::ai::jev_head::{HIDDEN, JevHead, QK_DIM};
use zorogravity_core::ai::model_store::model_dir;
use zorogravity_core::ai::verify::det_normal;

fn flat(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?)
}

/// (평균, 표준편차, 최솟값, 최댓값)
fn stats(v: &[f32]) -> (f32, f32, f32, f32) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, var.sqrt(), min, max)
}

fn frob(t: &Tensor) -> Result<f32> {
    Ok(t.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt())
}

/// 최대 특이값 (거듭제곱법)
fn top_singular(w: &Tensor) -> Result<f32> {
    let (_, cols) = w.dims2()?;
    let mut v = Tensor::from_vec(det_normal(cols, 99), (cols, 1), &Device::Cpu)?;
    v = v.affine(1.0 / frob(&v)? as f64, 0.0)?;
    for _ in 0..50 {
        let u = w.matmul(&v)?;
        let next = w.t()?.matmul(&u)?;
        v = next.affine(1.0 / frob(&next)? as f64, 0.0)?;
    }
    frob(&w.matmul(&v)?)
}

fn final_norm_weight(dir: &Path) -> Result<Vec<f32>> {
    let name = "model.language_model.norm.weight";
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json"))?)?;
    let file = index["weight_map"][name].as_str().context("index.json 에 최종 norm 이 없습니다")?;
    let tensors = candle_core::safetensors::load(dir.join(file), &Device::Cpu)?;
    flat(tensors.get(name).context("최종 norm 텐서를 읽지 못했습니다")?)
}

/// 행 i 마다 코사인 유사도 cos(a_i, b_i) 의 평균
fn mean_row_cosine(a: &Tensor, b: &Tensor) -> Result<f32> {
    let dot = (a * b)?.sum(1)?;
    let na = a.sqr()?.sum(1)?.sqrt()?;
    let nb = b.sqr()?.sum(1)?.sqrt()?;
    let denom = (&na * &nb)?;
    let cos = (&dot / &denom)?.to_vec1::<f32>()?;
    Ok(cos.iter().sum::<f32>() / cos.len() as f32)
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| model_dir("zoro").display().to_string());
    let dir = Path::new(&dir);
    let dev = Device::Cpu;
    let t = candle_core::safetensors::load(dir.join("jev_head.safetensors"), &dev)?;
    let get = |k: &str| -> Result<Tensor> { Ok(t.get(k).with_context(|| format!("{k} 없음"))?.to_dtype(DType::F32)?) };
    let (wq, bq, wk, bk) = (get("q.weight")?, get("q.bias")?, get("k.weight")?, get("k.bias")?);

    println!("== 1. 가중치 통계");
    for (name, tensor) in [("q.weight", &wq), ("k.weight", &wk), ("q.bias", &bq), ("k.bias", &bk)] {
        let (mean, std, min, max) = stats(&flat(tensor)?);
        println!(
            "{name:9} {:?}  평균 {mean:+.4}  표준편차 {std:.4}  범위 [{min:+.3}, {max:+.3}]  ‖·‖_F {:.3}",
            tensor.dims(),
            frob(tensor)?
        );
    }
    let init_row_norm = 1.0; // N(0, 1/2560) 로 초기화하면 행 노름은 약 1
    for (name, w) in [("q.weight", &wq), ("k.weight", &wk)] {
        let rows = flat(&w.sqr()?.sum(1)?.sqrt()?)?;
        let (m, s, lo, hi) = stats(&rows);
        let sigma = top_singular(w)?;
        println!(
            "{name:9} 행 노름 평균 {m:.3} (표준편차 {s:.3}, 범위 {lo:.3}~{hi:.3}; 초기값 기준 약 {init_row_norm:.1})  σ_max {sigma:.3}  안정 랭크 {:.1}",
            frob(w)?.powi(2) / sigma.powi(2)
        );
    }
    println!(
        "q 행과 k 행의 평균 코사인: {:+.4} (무작위면 0 근처)",
        mean_row_cosine(&wq, &wk)?
    );

    println!("\n== 2. 은닉 상태 규모 추정 (최종 norm 가중치)");
    let w = final_norm_weight(dir)?;
    let (wm, ws, wlo, whi) = stats(&w);
    let w_abs = w.iter().map(|x| x.abs()).sum::<f32>() / w.len() as f32;
    let w_norm = w.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("norm 가중치 평균 {wm:.3}  표준편차 {ws:.3}  범위 [{wlo:.2}, {whi:.2}]  |w| 평균 {w_abs:.3}  ‖w‖ {w_norm:.1}  (은닉 상태 노름 ≈ ‖w‖)");

    println!("\n== 3. 점수 규모 (h = w ⊙ z, z ~ N(0,1); 학습된 분포가 아니라 규모 어림)");
    let n = 512usize;
    let make = |seed: u64| -> Result<Tensor> {
        let z = det_normal(n * HIDDEN, seed);
        let h: Vec<f32> = z.chunks(HIDDEN).flat_map(|row| row.iter().zip(&w).map(|(z, w)| z * w).collect::<Vec<_>>()).collect();
        Ok(Tensor::from_vec(h, (n, HIDDEN), &dev)?)
    };
    let h1 = make(1)?;
    let h2 = make(2)?;
    let q = h1.matmul(&wq.t()?)?.broadcast_add(&bq)?;
    let k = h2.matmul(&wk.t()?)?.broadcast_add(&bk)?;
    let k_self = h1.matmul(&wk.t()?)?.broadcast_add(&bk)?;
    let s = flat(&(&q * &k)?.sum(1)?)?;
    let s_self = flat(&(&q * &k_self)?.sum(1)?)?;
    let scale = 1.0 / (QK_DIM as f32).sqrt();
    let (m, sd, _, _) = stats(&s);
    let (ms, sds, _, _) = stats(&s_self);
    println!("서로 다른 두 상태:  s 평균 {m:+.2} 표준편차 {sd:.2}   s/√256 평균 {:+.2} 표준편차 {:.2}", m * scale, sd * scale);
    println!("같은 상태 쌍:       s 평균 {ms:+.2} 표준편차 {sds:.2}   s/√256 평균 {:+.2} 표준편차 {:.2}", ms * scale, sds * scale);

    // 선택지 4개를 놓고 softmax 했을 때 최대 확률의 평균 (무작위 입력이므로 낮을수록 "뾰족하지 않음")
    let groups = n / 4;
    let qg = q.narrow(0, 0, groups)?.reshape((groups, 1, QK_DIM))?;
    let kg = k.narrow(0, 0, groups * 4)?.reshape((groups, 4, QK_DIM))?;
    let logits = qg.matmul(&kg.transpose(1, 2)?)?; // [groups, 1, 4]
    for (label, factor) in [("스케일 1/√256", scale), ("스케일 없음", 1.0)] {
        let p = softmax_last_dim(&logits.affine(factor as f64, 0.0)?)?.reshape((groups, 4))?;
        let top = flat(&p.max(1)?)?;
        let (mean_top, _, _, _) = stats(&top);
        let saturated = top.iter().filter(|x| **x > 0.99).count() as f32 / top.len() as f32;
        println!("{label:14} 선택지 4개의 최대 확률 평균 {mean_top:.3}, 0.99 초과 비율 {:.1}%", saturated * 100.0);
    }

    println!("\n== 4. 실제 가중치로 한 번 계산하는 시간 (CPU, F32)");
    let head = JevHead::from_safetensors(&dir.join("jev_head.safetensors"))?;
    let query = det_normal(HIDDEN, 5);
    let options = det_normal(4 * HIDDEN, 6);
    let warm = head.probabilities(&query, &options)?;
    let iters = 200;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        std::hint::black_box(head.probabilities(&query, &options)?);
    }
    println!(
        "선택지 4개: 호출당 {:.0} µs (확률 합 {:.4})",
        t.elapsed().as_secs_f64() * 1e6 / iters as f64,
        warm.iter().sum::<f32>()
    );
    Ok(())
}
