"""Concept formation over detector embeddings (Track D5): a CLASSIT/COBWEB-3-style concept hierarchy.

Cognitive-science grounding (details and references: plans/diffusion_cf_prompt.md, Addendum E):
  - Concepts are probabilistic prototypes: each node holds a diagonal Gaussian over the embedding
    dimensions (prototype theory; Posner & Keele 1968, Rosch 1975) plus the label counts of the
    instances it summarizes.
  - The hierarchy is formed incrementally by COBWEB's four operators (Fisher 1987) - add to the best
    child, create a new child, merge the two best children, split the best child - each step chosen to
    maximize category utility (Gluck & Corter 1985) in its continuous form (CLASSIT; Gennari, Langley &
    Fisher 1989): CU = (1/K) sum_k P(C_k) sum_i (1/sigma_ik - 1/sigma_ip) / (2 sqrt(pi)), with the
    standard deviations floored at an acuity.
  - The basic level (Rosch et al. 1976) is the depth whose partition has the highest category utility;
    we also report, per depth, the mutual information between concept and real/fake label - the
    distinctiveness whose peak marks the basic level in Wang, Singaravadivelan & MacLellan (2609.13047).
  - A clip is categorized by descending the tree (best child by CU, no update); its fake probability is
    the smoothed label distribution of the deepest concept on its path with at least `min_n` members.
"""
import math

import numpy as np

C = 1.0 / (2.0 * math.sqrt(math.pi))


class Node:
    __slots__ = ("n", "mean", "m2", "children", "labels", "id")

    def __init__(self, d, n_labels):
        self.n, self.mean, self.m2 = 0, np.zeros(d), np.zeros(d)
        self.children, self.labels, self.id = [], np.zeros(n_labels), -1

    def add(self, x, y):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)
        self.labels[y] += 1

    def absorb(self, other):
        """Merge another node's sufficient statistics into this one (parallel variance)."""
        n = self.n + other.n
        delta = other.mean - self.mean
        self.m2 = self.m2 + other.m2 + delta ** 2 * self.n * other.n / n
        self.mean = self.mean + delta * other.n / n
        self.n = n
        self.labels = self.labels + other.labels

    def copy(self):
        c = Node(len(self.mean), len(self.labels))
        c.n, c.mean, c.m2, c.labels = self.n, self.mean.copy(), self.m2.copy(), self.labels.copy()
        return c


def _inv_std(n, m2, acuity):
    var = np.where(n[..., None] > 0, m2 / np.maximum(n[..., None], 1), 0.0)
    return 1.0 / np.maximum(np.sqrt(var), acuity)


class ConceptTree:
    def __init__(self, dim, n_labels=2, acuity=0.1, max_depth=12, seed=0):
        self.d, self.L, self.acuity, self.max_depth = dim, n_labels, acuity, max_depth
        self.root = Node(dim, n_labels)
        self.rng = np.random.default_rng(seed)

    # ---- category utility of a candidate partition of `parent` (arrays over children) ----
    def _cu(self, pn, pm2, ns, m2s):
        """ns [K], m2s [K, d]; parent count/m2. Mean over children of P(C_k) * sum_i (1/s_ik - 1/s_ip)."""
        p_inv = _inv_std(np.array([pn]), pm2[None], self.acuity)[0]
        c_inv = _inv_std(ns, m2s, self.acuity)
        return float(((ns / pn)[:, None] * (c_inv - p_inv[None])).sum() * C / len(ns))

    @staticmethod
    def _stats_with(node, x):
        n = node.n + 1
        delta = x - node.mean
        mean = node.mean + delta / n
        return n, node.m2 + delta * (x - mean)

    def _evaluate(self, node, x):
        """CU of the four operators at `node` for instance x (parent statistics include x; nothing mutated)."""
        pn, pm2 = node.n + 1, self._stats_with(node, x)[1]
        kids = node.children
        ns = np.array([c.n for c in kids], float)
        m2s = np.stack([c.m2 for c in kids])
        best = []
        for i, c in enumerate(kids):  # add x to child i
            n2, m22 = self._stats_with(c, x)
            ns_i, m2s_i = ns.copy(), m2s.copy()
            ns_i[i], m2s_i[i] = n2, m22
            best.append(self._cu(pn, pm2, ns_i, m2s_i))
        order = np.argsort(best)[::-1]
        b1 = int(order[0])
        b2 = int(order[1]) if len(order) > 1 else None
        out = {"best": (best[b1], b1)}
        out["new"] = (self._cu(pn, pm2, np.append(ns, 1.0), np.vstack([m2s, np.zeros(self.d)])), None)
        if b2 is not None:  # merge b1 and b2, then add x
            m = kids[b1].copy()
            m.absorb(kids[b2])
            n2, m22 = self._stats_with(m, x)
            keep = [i for i in range(len(kids)) if i not in (b1, b2)]
            out["merge"] = (self._cu(pn, pm2, np.append(ns[keep], n2), np.vstack([m2s[keep], m22])), (b1, b2))
        if kids[b1].children:  # split b1 into its children, then add x to the best of them
            gk = kids[b1].children
            keep = [i for i in range(len(kids)) if i != b1]
            base_ns = np.concatenate([ns[keep], [g.n for g in gk]])
            base_m2 = np.vstack([m2s[keep]] + [g.m2 for g in gk])
            vals = []
            for j, g in enumerate(gk):
                n2, m22 = self._stats_with(g, x)
                nn, mm = base_ns.copy(), base_m2.copy()
                nn[len(keep) + j], mm[len(keep) + j] = n2, m22
                vals.append(self._cu(pn, pm2, nn, mm))
            out["split"] = (max(vals), b1)
        return out

    def ifit(self, x, y):
        """Incorporate one instance (COBWEB): choose an operator at each level, then update and descend."""
        node, depth = self.root, 0
        while True:
            if not node.children:
                if node.n > 0 and depth < self.max_depth:  # fringe split: the old leaf becomes a child
                    leaf = Node(self.d, self.L)
                    leaf.add(x, y)
                    node.children = [node.copy(), leaf]
                node.add(x, y)
                return
            ops = self._evaluate(node, x)
            op = max(ops, key=lambda k: ops[k][0])
            if depth >= self.max_depth:
                op = "best"
            if op == "split":  # replace the best child by its children, then decide again at this node
                gone = node.children.pop(ops["split"][1])
                node.children.extend(gone.children)
                continue
            node.add(x, y)
            if op == "best":
                node, depth = node.children[ops["best"][1]], depth + 1
            elif op == "new":
                leaf = Node(self.d, self.L)
                leaf.add(x, y)
                node.children.append(leaf)
                return
            else:  # merge the two best children under a new concept and descend into it
                b1, b2 = ops["merge"][1]
                a, b = node.children[b1], node.children[b2]
                m = a.copy()
                m.absorb(b)
                m.children = [a, b]
                node.children = [c for i, c in enumerate(node.children) if i not in (b1, b2)] + [m]
                node, depth = m, depth + 1

    def fit(self, X, Y, order=None):
        order = self.rng.permutation(len(X)) if order is None else order
        for i in order:
            self.ifit(X[i], int(Y[i]))
        self._number()
        return self

    def _number(self):
        k = 0
        stack = [self.root]
        while stack:
            n = stack.pop()
            n.id, k = k, k + 1
            stack.extend(n.children)

    def path(self, x):
        """Categorization without update: root -> ... best child by category utility."""
        node, out = self.root, [self.root]
        while node.children:
            kids = node.children
            ns = np.array([c.n for c in kids], float)
            m2s = np.stack([c.m2 for c in kids])
            pn, pm2 = node.n + 1, self._stats_with(node, x)[1]
            vals = []
            for i, c in enumerate(kids):
                n2, m22 = self._stats_with(c, x)
                nn, mm = ns.copy(), m2s.copy()
                nn[i], mm[i] = n2, m22
                vals.append(self._cu(pn, pm2, nn, mm))
            node = kids[int(np.argmax(vals))]
            out.append(node)
        return out

    def predict_proba(self, X, label=1, min_n=10, alpha=1.0):
        """P(label | deepest concept on the path with >= min_n members), Laplace-smoothed; also the node ids."""
        p, ids = np.zeros(len(X)), np.zeros(len(X), int)
        for i, x in enumerate(X):
            nodes = [n for n in self.path(x) if n.n >= min_n]
            n = nodes[-1]
            p[i] = (n.labels[label] + alpha) / (n.labels.sum() + alpha * self.L)
            ids[i] = n.id
        return p, ids

    # ---- basic level ----
    def levels(self, max_depth=None):
        """Per depth: the partition's category utility (relative to the root) and the size of the partition."""
        max_depth = max_depth or self.max_depth
        rows, frontier = [], [self.root]
        for depth in range(1, max_depth + 1):
            nxt = []
            for n in frontier:
                nxt.extend(n.children if n.children else [n])
            if len(nxt) == len(frontier):
                break
            ns = np.array([n.n for n in nxt], float)
            m2s = np.stack([n.m2 for n in nxt])
            lab = np.stack([n.labels for n in nxt])
            rows.append(dict(depth=depth, n_concepts=len(nxt), category_utility=self._cu(self.root.n, self.root.m2, ns, m2s),
                             label_mi=_mutual_info(lab)))
            frontier = nxt
        return rows


def _mutual_info(counts):
    """I(concept; label) in nats from a [concepts, labels] count table."""
    p = counts / counts.sum()
    pc, pl = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = p * np.log(p / (pc * pl))
    return float(np.nansum(t))
