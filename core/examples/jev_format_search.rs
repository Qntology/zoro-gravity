//! `jev_head` 를 학습한 형식 찾기: 명세의 정답 사례로 프롬프트 · 위치 조합을 모두 재 봅니다.
//!
//!   python scripts/extract_jev_cases.py zorogravity.txt out/jev_cases.json
//!   jev_format_search [사례.json] [zoro 폴더] [점수 파일]
//!
//! 형식(채팅 틀 · 본문)마다 순전파를 한 번 하고, 그 한 번의 은닉 상태로 질의 위치 · 선택지 위치 · 정규화 시점 조합을 모두 계산합니다.
//! 정확도는 정답 선택지가 1순위인 비율입니다. 소프트맥스 스케일은 순위를 바꾸지 않으므로 여기서 따지지 않습니다.
//! 선택지 순서의 영향을 빼려고 사례마다 정순 · 역순으로 두 번 묻습니다.
//! 조합을 많이 재면 우연히 잘 맞는 조합이 뽑히므로, 스케일 전 점수를 파일로 남겨 `scripts/jev_format_report.py` 가 교차 검증합니다.
//! 환경 변수 `ZOROGRAVITY_ONLY_BODY` 는 재는 본문 하나를, `ZOROGRAVITY_ONLY_WRAPPERS` 는 재는 틀 번호들(쉼표)을 정합니다.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::Result;
use candle_core::Device;
use zorogravity_core::ai::accel;
use zorogravity_core::ai::jev_head::JevHead;
use zorogravity_core::ai::model_store::model_dir;
use zorogravity_core::ai::residency::{Budget, Residency};
use zorogravity_core::ai::tokenizer::token_covering;
use zorogravity_core::ai::zoro::ZoroBackbone;

#[path = "shared/jev_format.rs"]
#[allow(dead_code)]
mod jev_format;
use jev_format::{render, Format, BEFORE, BODIES, NORMS, OPTION_POS, QUERIES, WRAPPERS};

#[derive(Clone)]
struct Case {
    id: String,
    gate: String,
    title: String,
    facts: Vec<(String, String)>,
    options: Vec<String>,
    answer: String,
    /// 명세 표의 기대 확률 (%)
    spec_pct: Option<f32>,
}

/// 명세의 사례에 더하는 대조 사례: 사실을 바꾸면 답이 달라지는지 봅니다. Gate 0 분류도 여기에 둡니다.
fn extra_cases() -> Vec<Case> {
    let g1: Vec<String> = ["G1_DOC_MISMATCH_REJECT", "G1_SIGN_VALID", "G1_SIGN_INVALID_RETRY", "G1_SIGN_ELDERLY_STAMP", "G1_EXPIRED_MANDATORY_REJECT", "G1_INTERCEPT_FORCE_HOLD"].iter().map(|s| s.to_string()).collect();
    let g3: Vec<String> = ["G3_ELIGIBLE_NORMAL", "G3_ELIGIBLE_HOMONYM_CLEAR", "G3_VIOLATION_HIGH_RISK", "G3_DENY_UNMASK_PRIVACY", "G3_INTERCEPT_RUBBER_STAMP"].iter().map(|s| s.to_string()).collect();
    let cats: Vec<String> = ["CAT_01_WELFARE", "CAT_02_SUBSIDY", "CAT_03_PERMIT", "CAT_04_CERT", "CAT_05_GRIEVANCE"].iter().map(|s| s.to_string()).collect();
    let f = |list: &[(&str, &str)]| -> Vec<(String, String)> { list.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect() };
    let mut out = vec![
        Case { id: "대조-서명정상".into(), gate: "G1".into(), title: "형식심사".into(), facts: f(&[("applicant_name", "김철수 (3음절)"), ("pad_stroke_count", "8회"), ("bbox_area_ratio", "42% (임계치 15% 초과)"), ("shape_feature", "MULTI_STROKE_SIGNATURE")]), options: g1.clone(), answer: "G1_SIGN_VALID".into(), spec_pct: None },
        Case { id: "대조-고령서명".into(), gate: "G1".into(), title: "형식심사".into(), facts: f(&[("applicant_age", "81세"), ("pad_stroke_count", "1회"), ("bbox_area_ratio", "2.1% (임계치 15% 미달)"), ("sign_retry_count", "2회 연속 실패"), ("shape_feature", "DOT_ONLY")]), options: g1, answer: "G1_SIGN_ELDERLY_STAMP".into(), spec_pct: None },
        Case { id: "대조-정상적격".into(), gate: "G3".into(), title: "실질 요건 심사".into(), facts: f(&[("landlord_name", "최민수"), ("mother_name", "박영희"), ("hash_match", "FALSE"), ("cohabit_history", "0건"), ("lattice_path", "없음"), ("gate2_transfer", "500,000원 전액 정상 이체")]), options: g3, answer: "G3_ELIGIBLE_NORMAL".into(), spec_pct: None },
    ];
    let titles: [(&str, &str); 5] = [
        ("[청년월세 지원 신청서, 주택 임대차계약서, 3개월 이체내역서, 금융정보 동의서]", "CAT_01_WELFARE"),
        ("[보조사업 정산보고서 총괄표, 카드매출전표 12매, 세금계산서 2매, 단체 통장사본]", "CAT_02_SUBSIDY"),
        ("[일반음식점 영업신고서, 위생교육필증, 건축물대장]", "CAT_03_PERMIT"),
        ("[인감증명 위임장, 위임자 신분증 사본, 대리인 신분증]", "CAT_04_CERT"),
        ("[자필 진정서 2매 (정형 표 없음, 도로공사 야간 소음 피해)]", "CAT_05_GRIEVANCE"),
    ];
    for (doc, answer) in titles {
        out.push(Case { id: format!("분류-{answer}"), gate: "G0".into(), title: "민원 분류".into(), facts: f(&[("scanned_titles", doc)]), options: cats.clone(), answer: answer.into(), spec_pct: None });
    }
    out
}

fn load(path: &str) -> Result<Vec<Case>> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let mut out = Vec::new();
    for c in v.as_array().unwrap() {
        out.push(Case {
            id: format!("명세-{}", c["id"].as_str().unwrap()),
            gate: c["gate"].as_str().unwrap().into(),
            title: c["title"].as_str().unwrap().into(),
            facts: c["facts"].as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().to_string(), p[1].as_str().unwrap().to_string())).collect(),
            options: c["options"].as_array().unwrap().iter().map(|o| o.as_str().unwrap().to_string()).collect(),
            answer: c["answer"].as_str().unwrap().into(),
            spec_pct: c["spec_pct"].as_f64().map(|v| v as f32),
        });
    }
    out.extend(extra_cases());
    Ok(out)
}

fn mean_rows(rows: &[&[f32]]) -> Vec<f32> {
    let mut out = vec![0f32; rows[0].len()];
    for r in rows {
        out.iter_mut().zip(*r).for_each(|(o, v)| *o += v);
    }
    out.iter_mut().for_each(|o| *o /= rows.len() as f32);
    out
}

#[derive(Default, Clone)]
struct Tally {
    hit: usize,
    total: usize,
    by_gate: BTreeMap<String, (usize, usize)>,
    /// 맞힌 건의 정답 확률 합
    conf: f32,
    misses: Vec<String>,
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "out/jev_cases.json".into());
    let dir = std::env::args().nth(2).unwrap_or_else(|| model_dir("zoro").display().to_string());
    let dump = std::env::args().nth(3).unwrap_or_else(|| "out/jev_scores.txt".into());
    let device = accel::select_device()?;
    println!("장치 {}", accel::backend_name(&device));
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    let backbone = ZoroBackbone::open(Path::new(&dir), residency)?;
    let head = JevHead::from_safetensors(&Path::new(&dir).join("jev_head.safetensors"))?;
    let cases = load(&path)?;
    println!("사례 {}건 · 형식 {} · 순전파 {}번", cases.len(), WRAPPERS.len() * BODIES.len(), WRAPPERS.len() * BODIES.len() * cases.len() * 2);

    // 틀 본문 질의 선택지 정규화 사례 순서 정답 점수…
    let mut dump = std::io::BufWriter::new(std::fs::File::create(dump)?);
    for (i, c) in cases.iter().enumerate() {
        writeln!(dump, "# {i} {} {} {}", c.id, c.gate, c.spec_pct.unwrap_or(-1.0))?;
    }

    let t = std::time::Instant::now();
    let mut table: BTreeMap<(usize, usize, usize, usize, usize), Tally> = BTreeMap::new();
    let tk = backbone.tokenizer();
    let only_body: Option<usize> = std::env::var("ZOROGRAVITY_ONLY_BODY").ok().and_then(|v| v.parse().ok());
    let only_wrappers: Option<Vec<usize>> = std::env::var("ZOROGRAVITY_ONLY_WRAPPERS").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect());
    for (w, wrapper) in WRAPPERS.iter().enumerate() {
        if only_wrappers.as_ref().is_some_and(|l| !l.contains(&w)) {
            continue;
        }
        for (b, body) in BODIES.iter().enumerate() {
            if only_body.is_some_and(|o| o != b) {
                continue;
            }
            for (ci, c) in cases.iter().enumerate() {
                for order in 0..2 {
                    let mut options = c.options.clone();
                    if order == 1 {
                        options.reverse();
                    }
                    let want = options.iter().position(|o| *o == c.answer).expect("정답 선택지");
                    let refs: Vec<&str> = options.iter().map(String::as_str).collect();
                    let r = render(w, b, &c.title, &c.facts, &refs);
                    let enc = tk.encode(&r.text)?;
                    let (pre, post) = backbone.model().hidden_states_pair(&enc.ids)?;
                    let d = post.dim(1)?;
                    let post = post.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                    let pre = pre.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                    for (n, hidden) in [&post, &pre].iter().enumerate() {
                        let row = |i: usize| &hidden[i * d..(i + 1) * d];
                        for (q, byte) in r.queries.iter().enumerate() {
                            let Some(qi) = token_covering(&enc.offsets, *byte) else { continue };
                            for o in 0..OPTION_POS.len() {
                                let mut keys = Vec::with_capacity(options.len() * d);
                                for &(s, e) in &r.spans {
                                    let v: Vec<f32> = match o {
                                        0 => row(token_covering(&enc.offsets, e - 1).unwrap()).to_vec(),
                                        1 => row(token_covering(&enc.offsets, s).unwrap()).to_vec(),
                                        2 => {
                                            let (a, z) = (token_covering(&enc.offsets, s).unwrap(), token_covering(&enc.offsets, e - 1).unwrap());
                                            mean_rows(&(a..=z).map(row).collect::<Vec<_>>())
                                        }
                                        3 => row(token_covering(&enc.offsets, s - 1).unwrap()).to_vec(),
                                        _ => row(token_covering(&enc.offsets, e).unwrap()).to_vec(),
                                    };
                                    keys.extend_from_slice(&v);
                                }
                                let scores = head.scores(row(qi), &keys)?;
                                let line: Vec<String> = scores.iter().map(|v| format!("{v:.4}")).collect();
                                writeln!(dump, "{w} {b} {q} {o} {n} {ci} {order} {want} {}", line.join(" "))?;
                                let p = head.probabilities(row(qi), &keys)?;
                                let pick = p.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|x| x.0).unwrap();
                                let e = table.entry((w, b, q, o, n)).or_default();
                                e.total += 1;
                                let g = e.by_gate.entry(c.gate.clone()).or_default();
                                g.1 += 1;
                                if pick == want {
                                    e.hit += 1;
                                    e.conf += p[want];
                                    g.0 += 1;
                                } else if order == 0 {
                                    e.misses.push(format!("{}→{}", c.id, options[pick]));
                                }
                            }
                        }
                    }
                }
            }
            println!("[{:.0} s] {wrapper} · {body} 끝", t.elapsed().as_secs_f64());
        }
    }

    let chance: f32 = cases.iter().map(|c| 1.0 / c.options.len() as f32).sum::<f32>() / cases.len() as f32;
    println!("\n우연히 맞히는 비율 {:.1}%", chance * 100.0);
    let mut rows: Vec<_> = table.iter().collect();
    rows.sort_by_key(|(_, e)| std::cmp::Reverse((e.hit, (e.conf * 1000.0) as i64)));
    println!("\n상위 조합");
    for ((w, b, q, o, n), e) in rows.iter().take(25) {
        let gates: Vec<String> = e.by_gate.iter().map(|(g, v)| format!("{g} {}/{}", v.0, v.1)).collect();
        println!("{:>3}/{:<3} ({:>5.1}%) 평균확률 {:.3} · {} · {} · 질의 {} · 선택지 {} · {} · {}", e.hit, e.total, 100.0 * e.hit as f32 / e.total as f32, e.conf / e.hit.max(1) as f32, WRAPPERS[*w], BODIES[*b], QUERIES[*q], OPTION_POS[*o], NORMS[*n], gates.join(" "));
    }
    dump.flush()?;
    let Format { wrapper, body, query, option, norm } = BEFORE;
    if let Some(now) = table.get(&(wrapper, body, query, option, norm)) {
        println!("\n탐색 전 조합: {}/{} ({:.1}%)", now.hit, now.total, 100.0 * now.hit as f32 / now.total as f32);
    }
    for ((w, b, q, o, n), e) in rows.iter().take(3) {
        println!("\n[{}·{}·{}·{}·{}] 틀린 사례 (정순): {}", WRAPPERS[*w], BODIES[*b], QUERIES[*q], OPTION_POS[*o], NORMS[*n], e.misses.join(", "));
    }
    // 축별 평균으로 어느 선택이 영향이 큰지 봅니다.
    println!("\n축별 평균 정확도");
    let axis = |name: &str, labels: &[&str], key: fn(&(usize, usize, usize, usize, usize)) -> usize| {
        let mut sum = vec![(0usize, 0usize); labels.len()];
        for (k, e) in &table {
            sum[key(k)].0 += e.hit;
            sum[key(k)].1 += e.total;
        }
        let s: Vec<String> = labels.iter().zip(&sum).map(|(l, (h, t))| format!("{l} {:.1}%", 100.0 * *h as f32 / *t as f32)).collect();
        println!("  {name}: {}", s.join(" · "));
    };
    axis("틀", &WRAPPERS, |k| k.0);
    axis("본문", &BODIES, |k| k.1);
    axis("질의", &QUERIES, |k| k.2);
    axis("선택지", &OPTION_POS, |k| k.3);
    axis("정규화", &NORMS, |k| k.4);
    accel::finish_process(&device, 0);
    Ok(())
}
