//! Min-cut / max-flow on a graph with terminal links (Boykov & Kolmogorov, 2004).
//!
//! Written from the published algorithm. Nodes are connected to a source and a sink with
//! `add_terminal`, to each other with `add_edge`; after `max_flow`, `in_source_segment` tells
//! which side of the minimum cut a node ends up on.

const NONE: i32 = -1;
const TERMINAL: i32 = -2;
const ORPHAN: i32 = -3;
const NIL: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Node {
    /// First outgoing arc.
    first: u32,
    /// Arc towards the parent, or `NONE` / `TERMINAL` / `ORPHAN`.
    parent: i32,
    /// Remaining capacity of the link to the source (> 0) or sink (< 0).
    tr_cap: f64,
    ts: u32,
    dist: u32,
    is_sink: bool,
    in_queue: bool,
}

#[derive(Clone, Copy)]
struct Arc {
    head: u32,
    next: u32,
    r_cap: f64,
}

pub struct Graph {
    nodes: Vec<Node>,
    /// Arcs come in pairs: the reverse (sister) of arc `a` is `a ^ 1`.
    arcs: Vec<Arc>,
    flow: f64,
    time: u32,
    queue: std::collections::VecDeque<u32>,
    orphans: std::collections::VecDeque<u32>,
}

impl Graph {
    pub fn new(node_count: usize, edge_hint: usize) -> Self {
        Self {
            nodes: vec![Node { first: NIL, parent: NONE, tr_cap: 0.0, ts: 0, dist: 0, is_sink: false, in_queue: false }; node_count],
            arcs: Vec::with_capacity(edge_hint * 2),
            flow: 0.0,
            time: 0,
            queue: std::collections::VecDeque::new(),
            orphans: std::collections::VecDeque::new(),
        }
    }

    /// Capacity from the source to `i` and from `i` to the sink.
    pub fn add_terminal(&mut self, i: usize, from_source: f64, to_sink: f64) {
        let n = &mut self.nodes[i];
        // The part that can go source -> i -> sink directly is already flow.
        self.flow += from_source.min(to_sink);
        n.tr_cap += from_source - to_sink;
    }

    /// Undirected-style edge: capacity `cap` from `i` to `j` and `rev_cap` back.
    pub fn add_edge(&mut self, i: usize, j: usize, cap: f64, rev_cap: f64) {
        let a = self.arcs.len() as u32;
        self.arcs.push(Arc { head: j as u32, next: self.nodes[i].first, r_cap: cap });
        self.arcs.push(Arc { head: i as u32, next: self.nodes[j].first, r_cap: rev_cap });
        self.nodes[i].first = a;
        self.nodes[j].first = a + 1;
    }

    fn set_active(&mut self, i: u32) {
        let n = &mut self.nodes[i as usize];
        if !n.in_queue {
            n.in_queue = true;
            self.queue.push_back(i);
        }
    }

    fn next_active(&mut self) -> Option<u32> {
        while let Some(i) = self.queue.pop_front() {
            self.nodes[i as usize].in_queue = false;
            if self.nodes[i as usize].parent != NONE {
                return Some(i);
            }
        }
        None
    }

    fn orphan_front(&mut self, i: u32) {
        self.nodes[i as usize].parent = ORPHAN;
        self.orphans.push_front(i);
    }

    fn orphan_rear(&mut self, i: u32) {
        self.nodes[i as usize].parent = ORPHAN;
        self.orphans.push_back(i);
    }

    /// Total flow, which equals the cost of the minimum cut.
    pub fn max_flow(&mut self) -> f64 {
        for i in 0..self.nodes.len() as u32 {
            let n = &mut self.nodes[i as usize];
            if n.tr_cap != 0.0 {
                n.is_sink = n.tr_cap < 0.0;
                n.parent = TERMINAL;
                n.ts = 0;
                n.dist = 1;
                self.set_active(i);
            } else {
                n.parent = NONE;
            }
        }
        let mut current: Option<u32> = None;
        loop {
            let i = match current.take().filter(|&c| self.nodes[c as usize].parent != NONE) {
                Some(c) => c,
                None => match self.next_active() {
                    Some(c) => c,
                    None => break,
                },
            };
            let mut middle: Option<u32> = None;
            let (is_sink, ts, dist) = {
                let n = &self.nodes[i as usize];
                (n.is_sink, n.ts, n.dist)
            };
            let mut a = self.nodes[i as usize].first;
            while a != NIL {
                let arc = self.arcs[a as usize];
                let usable = if is_sink { self.arcs[(a ^ 1) as usize].r_cap > 0.0 } else { arc.r_cap > 0.0 };
                if usable {
                    let j = arc.head;
                    let (jp, jsink, jts, jdist) = {
                        let n = &self.nodes[j as usize];
                        (n.parent, n.is_sink, n.ts, n.dist)
                    };
                    if jp == NONE {
                        let n = &mut self.nodes[j as usize];
                        n.is_sink = is_sink;
                        n.parent = (a ^ 1) as i32;
                        n.ts = ts;
                        n.dist = dist + 1;
                        self.set_active(j);
                    } else if jsink != is_sink {
                        middle = Some(if is_sink { a ^ 1 } else { a });
                        break;
                    } else if jts <= ts && jdist > dist {
                        let n = &mut self.nodes[j as usize];
                        n.parent = (a ^ 1) as i32;
                        n.ts = ts;
                        n.dist = dist + 1;
                    }
                }
                a = arc.next;
            }
            self.time += 1;
            if let Some(m) = middle {
                current = Some(i);
                self.augment(m);
                while let Some(o) = self.orphans.pop_front() {
                    if self.nodes[o as usize].is_sink {
                        self.adopt_sink(o);
                    } else {
                        self.adopt_source(o);
                    }
                }
            }
        }
        self.flow
    }

    /// Push the bottleneck flow along source -> ... -> `middle` -> ... -> sink.
    fn augment(&mut self, middle: u32) {
        let sister = middle ^ 1;
        let mut bottleneck = self.arcs[middle as usize].r_cap;

        // Source side: walk from the tail of `middle` up to the terminal.
        let mut i = self.arcs[sister as usize].head;
        loop {
            let p = self.nodes[i as usize].parent;
            if p == TERMINAL {
                break;
            }
            let a = p as u32;
            bottleneck = bottleneck.min(self.arcs[(a ^ 1) as usize].r_cap);
            i = self.arcs[a as usize].head;
        }
        bottleneck = bottleneck.min(self.nodes[i as usize].tr_cap);
        // Sink side.
        let mut k = self.arcs[middle as usize].head;
        loop {
            let p = self.nodes[k as usize].parent;
            if p == TERMINAL {
                break;
            }
            let a = p as u32;
            bottleneck = bottleneck.min(self.arcs[a as usize].r_cap);
            k = self.arcs[a as usize].head;
        }
        bottleneck = bottleneck.min(-self.nodes[k as usize].tr_cap);

        self.arcs[sister as usize].r_cap += bottleneck;
        self.arcs[middle as usize].r_cap -= bottleneck;

        let mut i = self.arcs[sister as usize].head;
        loop {
            let p = self.nodes[i as usize].parent;
            if p == TERMINAL {
                break;
            }
            let a = p as u32;
            self.arcs[a as usize].r_cap += bottleneck;
            self.arcs[(a ^ 1) as usize].r_cap -= bottleneck;
            let next = self.arcs[a as usize].head;
            if self.arcs[(a ^ 1) as usize].r_cap <= 0.0 {
                self.orphan_front(i);
            }
            i = next;
        }
        self.nodes[i as usize].tr_cap -= bottleneck;
        if self.nodes[i as usize].tr_cap <= 0.0 {
            self.orphan_front(i);
        }

        let mut k = self.arcs[middle as usize].head;
        loop {
            let p = self.nodes[k as usize].parent;
            if p == TERMINAL {
                break;
            }
            let a = p as u32;
            self.arcs[(a ^ 1) as usize].r_cap += bottleneck;
            self.arcs[a as usize].r_cap -= bottleneck;
            let next = self.arcs[a as usize].head;
            if self.arcs[a as usize].r_cap <= 0.0 {
                self.orphan_front(k);
            }
            k = next;
        }
        self.nodes[k as usize].tr_cap += bottleneck;
        if self.nodes[k as usize].tr_cap >= 0.0 {
            self.orphan_front(k);
        }
        self.flow += bottleneck;
    }

    /// Length of the path from `j` to its terminal if it is still rooted, else `None`.
    /// Caches distances with the current `time` stamp.
    fn root_distance(&mut self, mut j: u32) -> Option<u32> {
        let mut d = 0u32;
        loop {
            let n = self.nodes[j as usize];
            if n.ts == self.time {
                d += n.dist;
                break;
            }
            d += 1;
            match n.parent {
                TERMINAL => {
                    let m = &mut self.nodes[j as usize];
                    m.ts = self.time;
                    m.dist = 1;
                    break;
                }
                ORPHAN | NONE => return None,
                p => j = self.arcs[p as usize].head,
            }
        }
        Some(d)
    }

    /// Stamp the path from `j` with decreasing distances starting at `d`.
    fn stamp_path(&mut self, mut j: u32, mut d: u32) {
        while self.nodes[j as usize].ts != self.time {
            let n = &mut self.nodes[j as usize];
            n.ts = self.time;
            n.dist = d;
            d = d.saturating_sub(1);
            let p = n.parent;
            j = self.arcs[p as usize].head;
        }
    }

    fn adopt_source(&mut self, i: u32) {
        let mut best: Option<(u32, u32)> = None; // (arc, distance)
        let mut a0 = self.nodes[i as usize].first;
        while a0 != NIL {
            let arc = self.arcs[a0 as usize];
            if self.arcs[(a0 ^ 1) as usize].r_cap > 0.0 {
                let j = arc.head;
                let n = self.nodes[j as usize];
                if !n.is_sink && n.parent != NONE {
                    if let Some(d) = self.root_distance(j) {
                        if best.is_none_or(|(_, bd)| d < bd) {
                            best = Some((a0, d));
                        }
                        self.stamp_path(j, d);
                    }
                }
            }
            a0 = arc.next;
        }
        self.finish_adoption(i, best, false);
    }

    fn adopt_sink(&mut self, i: u32) {
        let mut best: Option<(u32, u32)> = None;
        let mut a0 = self.nodes[i as usize].first;
        while a0 != NIL {
            let arc = self.arcs[a0 as usize];
            if arc.r_cap > 0.0 {
                let j = arc.head;
                let n = self.nodes[j as usize];
                if n.is_sink && n.parent != NONE {
                    if let Some(d) = self.root_distance(j) {
                        if best.is_none_or(|(_, bd)| d < bd) {
                            best = Some((a0, d));
                        }
                        self.stamp_path(j, d);
                    }
                }
            }
            a0 = arc.next;
        }
        self.finish_adoption(i, best, true);
    }

    fn finish_adoption(&mut self, i: u32, best: Option<(u32, u32)>, sink: bool) {
        if let Some((a, d)) = best {
            let n = &mut self.nodes[i as usize];
            n.parent = a as i32;
            n.ts = self.time;
            n.dist = d + 1;
            return;
        }
        self.nodes[i as usize].parent = NONE;
        self.nodes[i as usize].ts = 0;
        let mut a0 = self.nodes[i as usize].first;
        while a0 != NIL {
            let arc = self.arcs[a0 as usize];
            let j = arc.head;
            let n = self.nodes[j as usize];
            if n.is_sink == sink && n.parent != NONE {
                let flows_to_i = if sink { arc.r_cap > 0.0 } else { self.arcs[(a0 ^ 1) as usize].r_cap > 0.0 };
                if flows_to_i {
                    self.set_active(j);
                }
                if n.parent >= 0 && self.arcs[n.parent as usize].head == i {
                    self.orphan_rear(j);
                }
            }
            a0 = arc.next;
        }
    }

    /// After `max_flow`: is `i` on the source side of the minimum cut?
    pub fn in_source_segment(&self, i: usize) -> bool {
        let n = &self.nodes[i];
        n.parent == NONE || !n.is_sink
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: Edmonds-Karp on a dense capacity matrix. Node `n` = source, `n + 1` = sink.
    fn reference(n: usize, cap: &[Vec<f64>]) -> f64 {
        let (s, t) = (n, n + 1);
        let mut c = cap.to_vec();
        let mut flow = 0.0;
        loop {
            let mut prev = vec![usize::MAX; n + 2];
            prev[s] = s;
            let mut q = std::collections::VecDeque::from([s]);
            while let Some(u) = q.pop_front() {
                for v in 0..n + 2 {
                    if prev[v] == usize::MAX && c[u][v] > 1e-12 {
                        prev[v] = u;
                        q.push_back(v);
                    }
                }
            }
            if prev[t] == usize::MAX {
                return flow;
            }
            let mut b = f64::MAX;
            let mut v = t;
            while v != s {
                b = b.min(c[prev[v]][v]);
                v = prev[v];
            }
            v = t;
            while v != s {
                c[prev[v]][v] -= b;
                c[v][prev[v]] += b;
                v = prev[v];
            }
            flow += b;
        }
    }

    #[test]
    fn two_nodes_simple_chain() {
        // s -> 0 (5), 0 -> 1 (3), 1 -> t (4): bottleneck 3.
        let mut g = Graph::new(2, 1);
        g.add_terminal(0, 5.0, 0.0);
        g.add_terminal(1, 0.0, 4.0);
        g.add_edge(0, 1, 3.0, 0.0);
        assert!((g.max_flow() - 3.0).abs() < 1e-9);
        assert!(g.in_source_segment(0));
        assert!(!g.in_source_segment(1));
    }

    #[test]
    fn direct_terminal_flow_counts() {
        let mut g = Graph::new(1, 0);
        g.add_terminal(0, 7.0, 2.0);
        assert!((g.max_flow() - 2.0).abs() < 1e-9);
        assert!(g.in_source_segment(0));
    }

    #[test]
    fn matches_edmonds_karp_on_random_graphs() {
        let mut seed = 12345u64;
        let mut rnd = |m: u64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) % m
        };
        for case in 0..60 {
            let n = 4 + (rnd(10) as usize);
            let mut cap = vec![vec![0.0; n + 2]; n + 2];
            let mut g = Graph::new(n, 64);
            for i in 0..n {
                let (a, b) = (rnd(6) as f64 * (rnd(2) as f64), rnd(6) as f64 * (rnd(2) as f64));
                g.add_terminal(i, a, b);
                cap[n][i] += a;
                cap[i][n + 1] += b;
            }
            for _ in 0..(n * 2) {
                let (i, j) = (rnd(n as u64) as usize, rnd(n as u64) as usize);
                if i == j {
                    continue;
                }
                let (f, r) = (rnd(7) as f64, rnd(7) as f64);
                g.add_edge(i, j, f, r);
                cap[i][j] += f;
                cap[j][i] += r;
            }
            let got = g.max_flow();
            let want = reference(n, &cap);
            assert!((got - want).abs() < 1e-9, "case {case}: {got} vs {want}");
        }
    }

    #[test]
    fn grid_segmentation_cuts_between_seeds() {
        // 1 x 10 chain, left end tied to the source, right end to the sink, weak link in the middle.
        let n = 10;
        let mut g = Graph::new(n, n);
        g.add_terminal(0, 100.0, 0.0);
        g.add_terminal(n - 1, 0.0, 100.0);
        for i in 0..n - 1 {
            let w = if i == 4 { 1.0 } else { 10.0 };
            g.add_edge(i, i + 1, w, w);
        }
        assert!((g.max_flow() - 1.0).abs() < 1e-9);
        for i in 0..=4 {
            assert!(g.in_source_segment(i), "{i}");
        }
        for i in 5..n {
            assert!(!g.in_source_segment(i), "{i}");
        }
    }
}
