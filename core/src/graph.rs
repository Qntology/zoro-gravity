//! 비식별 관계망 질의 (명세 Table 3 의 멀티홉 탐색을 사건 범위에서 구현).
//!
//! 노드는 솔트 해시로만 구분합니다. 이름은 동명이인을 가려내는 데만 씁니다.

use std::collections::{HashMap, VecDeque};

use crate::case::{Link, Relation};

/// 임차 배제 규칙에 쓰는 "가까운 연결"인지 (명세 Part 2: 2촌 이내 친족, 같은 세대, 같은 계좌)
pub fn is_close_link(link: Link) -> bool {
    match link {
        Link::Family { degree } => degree <= 2,
        Link::Household { .. } | Link::SharedAccount => true,
    }
}

/// `from` 과 `to` 가 가까운 연결만 따라 `max_hops` 홉 안에서 이어지면, 노드 해시 경로를 돌려줍니다.
pub fn close_link_path(relations: &[Relation], from: &str, to: &str, max_hops: usize) -> Option<Vec<String>> {
    if from == to {
        return Some(vec![from.to_string()]);
    }
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for r in relations.iter().filter(|r| is_close_link(r.link)) {
        adj.entry(r.a.hash.as_str()).or_default().push(r.b.hash.as_str());
        adj.entry(r.b.hash.as_str()).or_default().push(r.a.hash.as_str());
    }

    let mut parent: HashMap<&str, &str> = HashMap::new();
    let mut depth: HashMap<&str, usize> = HashMap::new();
    let mut queue: VecDeque<&str> = VecDeque::new();
    depth.insert(from, 0);
    queue.push_back(from);

    while let Some(cur) = queue.pop_front() {
        let d = depth.get(cur).copied().unwrap_or(0);
        if d >= max_hops {
            continue;
        }
        for &next in adj.get(cur).into_iter().flatten() {
            if depth.contains_key(next) {
                continue;
            }
            depth.insert(next, d + 1);
            parent.insert(next, cur);
            if next == to {
                let mut path = vec![to.to_string()];
                let mut at = to;
                while let Some(&p) = parent.get(at) {
                    path.push(p.to_string());
                    at = p;
                }
                path.reverse();
                return Some(path);
            }
            queue.push_back(next);
        }
    }
    None
}

/// 해시가 `hash` 와 다른 노드 중에 이름이 `name` 과 같은 사람이 있으면 true (동명이인 후보).
/// 명세 Case 3-A: 같은 이름의 모친이 있어도 생년월일·해시가 다르면 다른 사람입니다.
pub fn shares_name_with_other_node(relations: &[Relation], hash: &str, name: &str) -> bool {
    relations
        .iter()
        .any(|r| [&r.a, &r.b].iter().any(|n| n.name == name && n.hash != hash))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::Node;

    fn node(hash: &str, name: &str) -> Node {
        Node { hash: hash.into(), name: name.into() }
    }

    fn rel(a: Node, b: Node, link: Link) -> Relation {
        Relation { a, b, link }
    }

    #[test]
    fn finds_a_parent_one_hop_away() {
        let rels = [rel(node("applicant", "김철수"), node("mother", "박영희"), Link::Family { degree: 1 })];
        let path = close_link_path(&rels, "applicant", "mother", 3).unwrap();
        assert_eq!(path, vec!["applicant", "mother"]);
    }

    #[test]
    fn finds_a_household_link_two_hops_away() {
        let rels = [
            rel(node("applicant", "김철수"), node("sibling", "김영수"), Link::Household { year: 2024 }),
            rel(node("sibling", "김영수"), node("owner", "박영희"), Link::Family { degree: 2 }),
        ];
        assert!(close_link_path(&rels, "applicant", "owner", 3).is_some());
        assert!(close_link_path(&rels, "applicant", "owner", 1).is_none(), "두 홉은 한도 1 을 넘습니다");
    }

    #[test]
    fn distant_family_does_not_count() {
        let rels = [rel(node("applicant", "김철수"), node("cousin", "이민수"), Link::Family { degree: 4 })];
        assert!(close_link_path(&rels, "applicant", "cousin", 3).is_none());
    }

    #[test]
    fn same_name_with_different_hash_is_a_homonym_candidate() {
        let rels = [rel(node("applicant", "김철수"), node("mother", "박영희"), Link::Family { degree: 1 })];
        assert!(shares_name_with_other_node(&rels, "landlord", "박영희"));
        assert!(!shares_name_with_other_node(&rels, "mother", "박영희"), "같은 사람이면 동명이인이 아닙니다");
    }
}
