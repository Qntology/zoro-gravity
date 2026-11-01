//! zoro 자기 추측 디코딩 평가: 실제 글쓰기 프롬프트를 추측의 양별로 써 보고, 글이 같은지와 시간 · 채택률을 잽니다.
//!
//!   speculative_eval [zoro 폴더] [추측 목록]
//!
//! 추측은 `마스크-탐색-폭` 을 `_` 로 이은 목록입니다 (기본 `0-0-0_1-0-3_0-2-3_1-2-3`). 첫 항목이 기준이고 나머지는 기준과 글이 같은지 봅니다.
//! 프롬프트는 `drafting` 의 지시문 그대로입니다.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::Result;
use zorogravity_core::ai::accel;
use zorogravity_core::ai::residency::{Budget, Residency};
use zorogravity_core::ai::speculative::Speculation;
use zorogravity_core::ai::zoro::{parse_speculation, Written, ZoroBackbone};
use zorogravity_core::drafting::{normalize_excuse, review_opinion, summarize_grievance, TextWriter};

const GRIEVANCES: [&str; 8] = [
    "저는 위 주소에 거주하는 주민으로서 인근 공사장의 소음으로 인한 피해를 진정합니다. 매일 아침 일곱 시부터 저녁 여덟 시까지 공사 소음이 이어져 일상생활이 어렵습니다. 구청에 두 차례 전화를 드렸지만 아직 아무런 조치가 없습니다. 방음벽을 설치해 주시기를 부탁드립니다.",
    "우리 동네 앞 도로에 지난달부터 지름 50센티미터 정도의 구멍이 패여 있습니다. 비가 오면 물이 고여 잘 보이지 않아 지난주에 오토바이 한 대가 넘어졌습니다. 빨리 포장 공사를 해 주십시오.",
    "아파트 옆 음식점에서 음식물 쓰레기를 내놓아 악취가 심합니다. 여름에는 파리가 들끓어 창문을 열 수 없습니다. 세 번이나 말씀드렸는데 고쳐지지 않습니다. 위생 점검을 나와 주세요.",
    "초등학교 앞 횡단보도에 신호등이 없어 아이들이 위험합니다. 지난 3월에 접촉사고가 2건 있었습니다. 신호등을 설치해 주시고 과속 단속 카메라도 달아 주시기 바랍니다.",
    "저는 일흔두 살 독거노인입니다. 기초연금으로 생활하는데 난방비가 석 달치 밀렸습니다. 긴급 생계비 지원을 받을 수 있는지 알려 주시고 도와주십시오.",
    "재산세 고지서를 받았는데 작년보다 12만 원이 더 나왔습니다. 면적이 잘못 적힌 것 같습니다. 고지 금액을 확인해 주시고 잘못되었으면 고쳐 주십시오.",
    "재개발 현장에 새벽부터 덤프트럭이 드나들어 먼지와 진동이 심합니다. 작업 시간을 지키도록 지도해 주십시오.",
    "오래된 옹벽이 기울어져 있어 장마철마다 불안합니다. 점검해 주시기 바랍니다.",
];

const EXCUSES: [&str; 4] = [
    "지난달에 보일러가 고장 나서 수리비 20만 원을 제가 먼저 내고 월세에서 깠어요",
    "수도관 공사비 30만 원을 뺐어요. 영수증도 있습니다",
    "차액은 매달 만나서 현금으로 직접 건넸습니다",
    "회사 일이 바빠서 은행에 갈 시간이 없었습니다",
];

#[derive(Default)]
struct Totals {
    tokens: usize,
    prefill_ms: f64,
    decode_ms: f64,
    reads: usize,
    drafted: usize,
    accepted: usize,
    same: usize,
    prompts: usize,
}

impl Totals {
    fn add(&mut self, w: &Written, same: bool) {
        self.tokens += w.tokens;
        self.prefill_ms += w.prefill_ms;
        self.decode_ms += w.decode_ms;
        self.reads += w.reads;
        self.drafted += w.drafted;
        self.accepted += w.accepted;
        self.same += same as usize;
        self.prompts += 1;
    }
}

/// 같은 프롬프트를 추측의 양마다 씁니다. 판단 경로로는 첫 항목의 글을 돌려줍니다.
struct Compare<'a> {
    zoro: &'a ZoroBackbone,
    plans: Vec<Speculation>,
    totals: Mutex<BTreeMap<usize, Totals>>,
}

impl TextWriter for Compare<'_> {
    fn complete(&self, system: &str, user: &str, max_tokens: usize) -> Result<String> {
        let want = self.zoro.write_with(system, user, max_tokens, self.plans[0])?;
        let mut totals = self.totals.lock().unwrap();
        totals.entry(0).or_default().add(&want, true);
        for (i, &plan) in self.plans.iter().enumerate().skip(1) {
            let got = self.zoro.write_with(system, user, max_tokens, plan)?;
            let same = got.text == want.text && got.finished == want.finished;
            if !same {
                println!("{plan:?} 의 글이 다릅니다\n  기준: {:?}\n  추측: {:?}", want.text, got.text);
            }
            totals.entry(i).or_default().add(&got, same);
        }
        Ok(want.text)
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args.first().cloned().unwrap_or_else(|| "models/zoro".into());
    let plans: Vec<Speculation> =
        args.get(1).map_or("0-0-0_1-0-3_0-2-3_1-2-3", String::as_str).split('_').map(|p| parse_speculation(&p.replace('-', ":"))).collect::<Result<_>>()?;

    let device = accel::select_device()?;
    println!("장치 {}", accel::backend_name(&device));
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    let zoro = ZoroBackbone::open(Path::new(&dir), residency)?;
    println!("기본 추측 {:?}", zoro.model().default_speculation());
    // 첫 호출은 가중치를 올리고 커널을 준비하므로 재지 않습니다.
    zoro.write_with("한국어로 답합니다.", "안녕하세요.", 4, Speculation::NONE)?;

    let compare = Compare { zoro: &zoro, plans: plans.clone(), totals: Mutex::new(BTreeMap::new()) };
    for text in GRIEVANCES {
        summarize_grievance(&compare, text)?;
    }
    for text in EXCUSES {
        normalize_excuse(&compare, text)?;
    }
    let facts: Vec<String> = ["처분: 정상 처리", "계약 월세: 500000원", "최근 이체액: 500000원"].map(String::from).to_vec();
    review_opinion(&compare, &facts)?;

    println!("\n마스크-탐색-폭 · 글 수 · 같은 글 · 토큰 · 프리필 ms · 토큰당 ms · 순전파 · 패스당 토큰 · 후보 채택");
    for (i, t) in compare.totals.lock().unwrap().iter() {
        println!(
            "{}-{}-{} · {} · {} · {} · {:.0} · {:.1} · {} · {:.2} · {}/{} ({:.1}%)",
            plans[*i].masks,
            plans[*i].lookup,
            plans[*i].width,
            t.prompts,
            t.same,
            t.tokens,
            t.prefill_ms / t.prompts.max(1) as f64,
            t.decode_ms / t.tokens.max(1) as f64,
            t.reads,
            t.tokens as f64 / t.reads.max(1) as f64,
            t.accepted,
            t.drafted,
            100.0 * t.accepted as f64 / t.drafted.max(1) as f64,
        );
    }
    accel::finish_process(&device, 0);
    Ok(())
}
