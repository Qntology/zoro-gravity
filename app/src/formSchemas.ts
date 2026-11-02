/** 민원 입력 항목. 키 이름은 코어 `forms.rs` 의 키와 같아야 합니다.
 *  민원인이 정하는 값만 받습니다. 계약 월세 · 이체액 · 필증 같은 값은 서류나 대장에서 옵니다. */
export type FieldKind = "text" | "birth" | "money" | "count" | "date" | "choice" | "account" | "tel" | "long";
export type CategoryKey = "welfare" | "subsidy" | "permit" | "cert" | "grievance";

export interface Field {
  key: string;
  label: string;
  kind: FieldKind;
  options?: string[];
  max?: number;
  hint?: string;
  optional?: boolean;
}

export interface CategorySchema {
  key: CategoryKey;
  label: string;
  /** 처음 화면의 한 줄 설명 */
  blurb: string;
  fields: Field[];
  /** 서명이 필요한 민원 */
  sign: boolean;
  /** 금융정보 조회 동의가 필요한 민원 */
  consent: boolean;
}

const applicant = (label: string): Field[] => [
  { key: "applicant_name", label: `${label} 성함`, kind: "text", max: 20 },
  { key: "applicant_birth", label: "생년월일 앞 6자리", kind: "birth" },
];

export const SCHEMAS: CategorySchema[] = [
  {
    key: "welfare",
    label: "복지 지원",
    blurb: "청년월세 지원을 신청합니다",
    fields: [
      ...applicant("신청인"),
      { key: "applied_rent", label: "신청 월세 (원)", kind: "money" },
      { key: "handwritten_account", label: "입금받을 계좌", kind: "account", hint: "숫자와 하이픈" },
    ],
    sign: true,
    consent: true,
  },
  {
    key: "subsidy",
    label: "지원금 정산",
    blurb: "보조사업 사용 내역을 정산합니다",
    fields: [
      ...applicant("대표자"),
      { key: "org_name", label: "단체명", kind: "text", max: 40 },
      { key: "program", label: "사업명", kind: "text", max: 40 },
      { key: "claimed_total", label: "정산 금액 (원)", kind: "money" },
    ],
    sign: false,
    consent: false,
  },
  {
    key: "permit",
    label: "허가 · 신고",
    blurb: "영업신고를 합니다",
    fields: [
      ...applicant("신고인"),
      { key: "business_name", label: "상호", kind: "text", max: 40 },
      { key: "business_type", label: "업종", kind: "choice", options: ["일반음식점", "휴게음식점", "제과점", "기타"] },
      { key: "address", label: "영업장 주소", kind: "text", max: 80 },
      { key: "area_m2", label: "면적 (㎡)", kind: "count" },
    ],
    sign: false,
    consent: false,
  },
  {
    key: "cert",
    label: "증명 · 대리",
    blurb: "위임장으로 대리 발급을 받습니다",
    fields: [
      ...applicant("대리인"),
      { key: "principal_name", label: "위임자 성함", kind: "text", max: 20 },
      { key: "principal_birth", label: "위임자 생년월일 앞 6자리", kind: "birth" },
      { key: "relation", label: "위임자와의 관계", kind: "choice", options: ["배우자", "자녀", "부모", "형제자매", "기타"] },
      { key: "document", label: "발급 종류", kind: "choice", options: ["인감증명서", "기타 증명서"] },
      { key: "purpose", label: "용도", kind: "text", max: 40 },
      { key: "poa_date", label: "위임장 작성일", kind: "date" },
    ],
    sign: true,
    consent: false,
  },
  {
    key: "grievance",
    label: "고충민원",
    blurb: "불편한 점을 담당 부서에 전합니다",
    fields: [
      ...applicant("민원인"),
      { key: "contact", label: "연락처", kind: "tel", optional: true, hint: "예: 010-1234-5678" },
      { key: "body", label: "내용 (10자 이상)", kind: "long", max: 4000 },
    ],
    sign: false,
    consent: false,
  },
];

export function schemaOf(key: CategoryKey): CategorySchema {
  const found = SCHEMAS.find((s) => s.key === key);
  if (!found) throw new Error(`알 수 없는 민원: ${key}`);
  return found;
}

/** 민원인이 고를 수 있는 소명. 차액이 있을 때만 보입니다. */
export const EXCUSES: { value: string; label: string }[] = [
  { value: "typo", label: "신청서 금액을 잘못 적었습니다" },
  { value: "cash", label: "차액을 현금으로 직접 드렸습니다" },
  { value: "repair_receipt", label: "수리비를 월세에서 뺐습니다 (영수증 있음)" },
  { value: "repair_no_receipt", label: "수리비를 월세에서 뺐습니다 (영수증 없음)" },
  { value: "free_text", label: "직접 적겠습니다" },
];
