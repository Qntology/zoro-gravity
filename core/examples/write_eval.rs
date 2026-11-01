//! zoro 글쓰기 평가: 합성 고충민원과 소명으로 요약 · 정규화 · 숫자 대조를 실제 모델로 잽니다.
//!
//!   write_eval [zoro 폴더]
//!
//! 민원은 제가 쓴 합성 글이고, 부서와 소명 코드의 정답은 글을 쓸 때 정했습니다. 요약의 내용 품질은 기계로 재지 못하므로 글을 그대로 출력합니다.
//! `GRIEVANCES` 는 지시문을 고치며 본 글이고, `FRESH` 는 고친 뒤에 처음 돌린 글입니다. 두 결과를 나눠서 봅니다.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use zorogravity_core::ai::accel;
use zorogravity_core::ai::model_store::model_dir;
use zorogravity_core::ai::residency::{Budget, Residency};
use zorogravity_core::ai::zoro::ZoroBackbone;
use zorogravity_core::case::Excuse;
use zorogravity_core::drafting::{disposition, normalize_excuse, review_opinion, summarize_grievance};

const GRIEVANCES: [(&str, &str); 10] = [
    ("건설관리과", "저는 위 주소에 거주하는 주민으로서 인근 공사장의 소음으로 인한 피해를 진정합니다. 매일 아침 일곱 시부터 저녁 여덟 시까지 공사 소음이 이어져 일상생활이 어렵습니다. 구청에 두 차례 전화를 드렸지만 아직 아무런 조치가 없습니다. 방음벽을 설치해 주시기를 부탁드립니다."),
    ("도로관리과", "우리 동네 앞 도로에 지난달부터 지름 50센티미터 정도의 구멍이 패여 있습니다. 비가 오면 물이 고여 잘 보이지 않아 지난주에 오토바이 한 대가 넘어졌습니다. 빨리 포장 공사를 해 주십시오."),
    ("환경위생과", "아파트 옆 음식점에서 음식물 쓰레기를 내놓아 악취가 심합니다. 여름에는 파리가 들끓어 창문을 열 수 없습니다. 세 번이나 말씀드렸는데 고쳐지지 않습니다. 위생 점검을 나와 주세요."),
    ("교통행정과", "초등학교 앞 횡단보도에 신호등이 없어 아이들이 위험합니다. 지난 3월에 접촉사고가 2건 있었습니다. 신호등을 설치해 주시고 과속 단속 카메라도 달아 주시기 바랍니다."),
    ("공원녹지과", "근린공원의 가로등 다섯 개가 석 달째 꺼져 있고 산책로 나무가 쓰러져 길을 막고 있습니다. 저녁에 어두워 어르신들이 넘어질까 걱정됩니다. 수리와 정비를 요청합니다."),
    ("청소행정과", "골목 입구에 불법 투기된 쓰레기가 일주일째 쌓여 있습니다. 수거 차량이 오지 않아 냄새가 납니다. 수거해 주시고 불법 투기 감시 카메라를 설치해 주세요."),
    ("복지정책과", "저는 일흔두 살 독거노인입니다. 기초연금으로 생활하는데 난방비가 석 달치 밀렸습니다. 긴급 생계비 지원을 받을 수 있는지 알려 주시고 도와주십시오."),
    ("주택과", "살고 있는 다세대주택 옥상이 불법으로 증축되어 비가 새고 있습니다. 집주인에게 말해도 고쳐 주지 않습니다. 현장 확인을 하시고 시정 명령을 내려 주세요."),
    ("세무과", "재산세 고지서를 받았는데 작년보다 12만 원이 더 나왔습니다. 면적이 잘못 적힌 것 같습니다. 고지 금액을 확인해 주시고 잘못되었으면 고쳐 주십시오."),
    ("안전총괄과", "하천 옆 축대에 금이 가고 흙이 쏟아지고 있습니다. 비가 많이 오면 무너질 것 같아 주민 30가구가 불안해합니다. 긴급 안전 점검을 해 주십시오."),
];

const FRESH: [(&str, &str); 6] = [
    ("건설관리과", "재개발 현장에 새벽부터 덤프트럭이 드나들어 먼지와 진동이 심합니다. 작업 시간을 지키도록 지도해 주십시오."),
    ("도로관리과", "집 앞 보도블록이 들떠서 어머니가 걸려 넘어지셨습니다. 보수해 주십시오."),
    ("안전총괄과", "오래된 옹벽이 기울어져 있어 장마철마다 불안합니다. 점검해 주시기 바랍니다."),
    ("환경위생과", "지하 식당 환풍구에서 기름 냄새가 올라와 머리가 아픕니다. 단속해 주십시오."),
    ("교통행정과", "버스 정류장 앞 불법 주차 때문에 버스가 서지 못합니다. 단속해 주세요."),
    ("세무과", "자동차세를 이미 냈는데 체납 안내문이 왔습니다. 확인해 주세요."),
];

const EXCUSES: [(&str, &str); 8] = [
    ("NO_RECEIPT", "지난달에 보일러가 고장 나서 수리비 20만 원을 제가 먼저 내고 월세에서 깠어요"),
    ("NO_RECEIPT", "싱크대 고치는 데 든 돈을 월세에서 빼기로 했습니다"),
    ("WITH_RECEIPT", "에어컨 수리비를 월세에서 공제했고 수리 영수증을 가지고 있습니다"),
    ("WITH_RECEIPT", "수도관 공사비 30만 원을 뺐어요. 영수증도 있습니다"),
    ("CASH", "모자란 20만 원은 집주인께 현금으로 드렸어요"),
    ("CASH", "차액은 매달 만나서 현금으로 직접 건넸습니다"),
    ("OTHER", "회사 일이 바빠서 은행에 갈 시간이 없었습니다"),
    ("OTHER", "왜 그런지 저도 잘 모르겠습니다"),
];

fn code(e: &Option<Excuse>) -> &'static str {
    match e {
        Some(Excuse::RepairDeduction { has_receipt: false }) => "NO_RECEIPT",
        Some(Excuse::RepairDeduction { has_receipt: true }) => "WITH_RECEIPT",
        Some(Excuse::CashPaidDirectly) => "CASH",
        _ => "OTHER",
    }
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| model_dir("zoro").display().to_string());
    let device = accel::select_device()?;
    println!("장치 {}", accel::backend_name(&device));
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    let zoro = ZoroBackbone::open(Path::new(&dir), residency)?;

    let t = Instant::now();
    let probe = zoro.write("한국어로 답합니다.", "대한민국의 수도는 어디입니까? 한 문장으로 답하세요.", 40)?;
    println!("시험: {:?} · 토큰 {} · 프리필 {:.0} ms · 토큰당 {:.0} ms · 전체 {:.1} s", probe.text, probe.tokens, probe.prefill_ms, probe.decode_ms / probe.tokens.max(1) as f64, t.elapsed().as_secs_f64());

    for (title, set) in [("고충민원 요약 · 지시문을 고치며 본 글", &GRIEVANCES[..]), ("고충민원 요약 · 처음 돌린 글", &FRESH[..])] {
        println!("\n[{title}]");
        let (mut verified, mut dept) = (0, 0);
        let t = Instant::now();
        for (want, text) in set {
            let s = summarize_grievance(&zoro, text)?;
            verified += s.verified as usize;
            dept += (s.department.as_deref() == Some(*want)) as usize;
            println!("정답 {want} · 부서 {:?} · 검증 {} {:?}\n  요지: {}\n  요구: {}", s.department, s.verified, s.problems, s.gist, s.demand);
        }
        println!("검증 통과 {verified}/{} · 부서 일치 {dept}/{} · 건당 {:.1} s", set.len(), set.len(), t.elapsed().as_secs_f64() / set.len() as f64);
    }

    println!("\n[소명 정규화]");
    let mut hit = 0;
    let t = Instant::now();
    for (want, text) in EXCUSES {
        let got = code(&normalize_excuse(&zoro, text)?);
        hit += (got == want) as usize;
        println!("{} 정답 {want} · 결과 {got} · {text}", if got == want { "맞음" } else { "틀림" });
    }
    println!("일치 {hit}/{} · 건당 {:.1} s", EXCUSES.len(), t.elapsed().as_secs_f64() / EXCUSES.len() as f64);

    println!("\n[심사 의견]");
    let cases: [(&str, &[&str]); 3] = [
        ("G3_ELIGIBLE_NORMAL", &["계약 월세: 500000원", "최근 이체액: 500000원"]),
        ("G3_VIOLATION_HIGH_RISK", &["판단 사유: 임대인과 신청인 사이에 가까운 연결 경로가 있습니다", "계약 월세: 500000원", "최근 이체액: 300000원"]),
        ("SUPPLEMENT_7DAYS", &["판단 사유: 금융정보동의서가 빠졌습니다"]),
    ];
    for (code, rest) in cases {
        let facts: Vec<String> = std::iter::once(format!("처분: {}", disposition(code))).chain(rest.iter().map(|s| s.to_string())).collect();
        println!("{:?}", review_opinion(&zoro, &facts)?);
    }
    accel::finish_process(&device, 0);
    Ok(())
}
