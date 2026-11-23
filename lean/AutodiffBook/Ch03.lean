import Mathlib

/-!
# Chapter 3: Computational graphs and forward mode

Machine-checked versions of results the chapter proves by hand, named by
their labels in `chapters/03-forward-mode.qmd`. Dual numbers are Mathlib's
`DualNumber ℝ`, written `ℝ[ε]`: pairs `(a, b)` written `a + bε`, with
`fst` and `snd` returning `a` and `b`.
-/

open Polynomial DualNumber TrivSqZeroExt

namespace AutodiffBook.Ch03

/-- `def-dual-numbers`: the product of dual numbers carries the product rule in
its `ε` part, `(a + bε)(c + dε) = ac + (ad + bc)ε`. -/
theorem dual_mul (x y : ℝ[ε]) :
    (x * y).fst = x.fst * y.fst ∧ (x * y).snd = x.fst * y.snd + x.snd * y.fst :=
  ⟨fst_mul x y, DualNumber.snd_mul x y⟩

/-- `def-dual-numbers`: `ε² = 0`. -/
theorem eps_sq : (ε * ε : ℝ[ε]) = 0 := eps_mul_eps

/-- The statement of `thm-dual-numbers` for one polynomial, split into its real
and `ε` parts. -/
def DualRule (x : ℝ[ε]) (p : ℝ[X]) : Prop :=
  (aeval x p).fst = p.eval x.fst ∧ (aeval x p).snd = p.derivative.eval x.fst * x.snd

lemma dualRule_add (x : ℝ[ε]) {p q : ℝ[X]} (hp : DualRule x p) (hq : DualRule x q) :
    DualRule x (p + q) := by
  unfold DualRule at *
  simp only [map_add, fst_add, snd_add, eval_add]
  constructor
  · rw [hp.1, hq.1]
  · rw [hp.2, hq.2]; ring

lemma dualRule_mul (x : ℝ[ε]) {p q : ℝ[X]} (hp : DualRule x p) (hq : DualRule x q) :
    DualRule x (p * q) := by
  unfold DualRule at *
  simp only [map_mul, fst_mul, DualNumber.snd_mul, eval_mul, derivative_mul, eval_add]
  constructor
  · rw [hp.1, hq.1]
  · rw [hp.1, hp.2, hq.1, hq.2]; ring

lemma dualRule_C (x : ℝ[ε]) (c : ℝ) : DualRule x (C c) := by
  unfold DualRule
  simp [aeval_C, derivative_C, algebraMap_eq_inl]

lemma dualRule_X (x : ℝ[ε]) : DualRule x X := by
  unfold DualRule
  simp [aeval_X, derivative_X]

/-- `thm-dual-numbers`, polynomial case: for every real polynomial `p`,
`p(a + bε) = p(a) + p'(a) b ε`. -/
theorem dual_polynomial (p : ℝ[X]) (x : ℝ[ε]) :
    (aeval x p).fst = p.eval x.fst ∧
      (aeval x p).snd = p.derivative.eval x.fst * x.snd := by
  induction p using Polynomial.induction_on with
  | C c => exact dualRule_C x c
  | add p q hp hq => exact dualRule_add x hp hq
  | monomial n c h =>
    rw [pow_succ, ← mul_assoc]
    exact dualRule_mul x h (dualRule_X x)

/-- `thm-dual-numbers`, in the form `p(a + bε) = p(a) + p'(a) b ε` with `a + bε`
written as `inl a + b • ε`. -/
theorem dual_polynomial' (p : ℝ[X]) (a b : ℝ) :
    aeval (inl a + b • ε : ℝ[ε]) p = inl (p.eval a) + (p.derivative.eval a * b) • ε := by
  obtain ⟨h1, h2⟩ := dual_polynomial p (inl a + b • ε)
  ext
  · simp [h1]
  · simp [h2]

/-- `thm-topological-order`: every finite directed acyclic graph has a topological
order. The graph has an edge from `a` to `b` when `r a b`; it is acyclic when no
path of one or more edges leads from a node back to itself. The conclusion
numbers the nodes `0, …, n - 1` so that every edge goes from a smaller number to
a larger one. -/
theorem exists_topological_order {α : Type*} [Fintype α] (r : α → α → Prop)
    (acyclic : ∀ a, ¬ Relation.TransGen r a a) :
    ∃ e : α ≃ Fin (Fintype.card α), ∀ a b, r a b → e a < e b := by
  classical
  -- Reachability is a partial order because the graph has no cycles.
  let R := Relation.ReflTransGen r
  have hpo : IsPartialOrder α R :=
    { refl := fun _ => Relation.ReflTransGen.refl
      trans := fun _ _ _ => Relation.ReflTransGen.trans
      antisymm := by
        intro a b hab hba
        by_contra hne
        rcases Relation.ReflTransGen.cases_head hab with h | ⟨c, hac, hcb⟩
        · exact hne h
        · exact acyclic a (Relation.TransGen.head' hac (hcb.trans hba)) }
  obtain ⟨s, hs, hRs⟩ := extend_partialOrder R
  let L : LinearOrder α :=
    { le := s
      le_refl := hs.refl
      le_trans := fun _ _ _ => hs.trans _ _ _
      le_antisymm := fun _ _ => hs.antisymm _ _
      le_total := hs.total
      toDecidableLE := Classical.decRel _ }
  let e := Fintype.orderIsoFinOfCardEq α (k := Fintype.card α) rfl
  refine ⟨e.symm.toEquiv, fun a b hab => ?_⟩
  have hne : a ≠ b := fun h => acyclic a (Relation.TransGen.single (h ▸ hab))
  have hle : a ≤ b := hRs a b (Relation.ReflTransGen.single hab)
  exact e.symm.lt_iff_lt.mpr (lt_of_le_of_ne hle hne)

end AutodiffBook.Ch03
