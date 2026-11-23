import Mathlib

/-!
# Chapter 8: Probability and loss functions

Machine-checked versions of results that chapter 8
(`chapters/08-probability-and-loss.qmd`) proves by hand. Distributions
on a finite set are functions `p : ι → ℝ` with non-negative values
summing to 1. Lean's `Real.log 0 = 0` makes `p i * log (p i / q i)`
equal 0 when `p i = 0`, which matches the book's convention `0 ln 0 = 0`.
-/

open Finset Real

namespace AutodiffBook.Ch08

variable {ι : Type*} [Fintype ι]

/-- `lem-prob-log-bound`: `ln x ≤ x - 1` for `x > 0`. -/
theorem log_le_sub_one {x : ℝ} (hx : 0 < x) : log x ≤ x - 1 :=
  log_le_sub_one_of_pos hx

/-- `lem-prob-log-bound`, the equality case: `ln x = x - 1` only at `x = 1`. -/
theorem eq_one_of_log_eq_sub_one {x : ℝ} (hx : 0 < x) (h : log x = x - 1) : x = 1 := by
  by_contra hne
  exact (log_lt_sub_one_of_pos hx hne).ne h

/-- One term of the proof of `thm-gibbs`: `p - q ≤ p ln (p/q)` for
`p ≥ 0` and `q > 0`. -/
theorem sub_le_mul_log_div {p q : ℝ} (hp : 0 ≤ p) (hq : 0 < q) :
    p - q ≤ p * log (p / q) := by
  rcases hp.eq_or_lt with rfl | hp
  · simp [hq.le]
  · have h := log_le_sub_one_of_pos (div_pos hq hp)
    rw [← neg_neg (log (p / q)), ← log_inv, inv_div]
    have : p * (q / p - 1) = q - p := by field_simp
    nlinarith

/-- `thm-gibbs`: the Kullback–Leibler divergence `∑ p i ln (p i / q i)` is
non-negative. (The book also allows `q i = 0`; then the divergence is
`+∞` when `p i > 0`, and there is nothing to prove.) -/
theorem gibbs (p q : ι → ℝ) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i)
    (hp1 : ∑ i, p i = 1) (hq1 : ∑ i, q i = 1) :
    0 ≤ ∑ i, p i * log (p i / q i) := by
  have h := sum_le_sum fun i (_ : i ∈ univ) => sub_le_mul_log_div (hp i) (hq i)
  rw [sum_sub_distrib, hp1, hq1, sub_self] at h
  exact h

/-- `thm-gibbs`, the equality case: the divergence is 0 only when `p = q`. -/
theorem gibbs_eq_zero_iff (p q : ι → ℝ) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i)
    (hp1 : ∑ i, p i = 1) (hq1 : ∑ i, q i = 1) :
    ∑ i, p i * log (p i / q i) = 0 ↔ p = q := by
  constructor
  · intro h0
    -- Each term `p ln (p/q) - (p - q)` is non-negative and they sum to 0.
    have hsum : ∑ i, (p i * log (p i / q i) - (p i - q i)) = 0 := by
      rw [sum_sub_distrib, sum_sub_distrib, h0, hp1, hq1]; ring
    have hterm := (sum_eq_zero_iff_of_nonneg fun i _ =>
      sub_nonneg.mpr (sub_le_mul_log_div (hp i) (hq i))).mp hsum
    funext i
    have hi := hterm i (mem_univ i)
    rcases (hp i).eq_or_lt with h | h
    · -- `p i = 0` would make the term equal to `q i > 0`.
      rw [← h] at hi; simp at hi; linarith [hq i]
    · have hr : log (q i / p i) = q i / p i - 1 := by
        have : p i * log (p i / q i) = p i - q i := by linarith
        rw [← inv_div, log_inv]
        field_simp
        linarith
      have := eq_one_of_log_eq_sub_one (div_pos (hq i) h) hr
      field_simp at this
      linarith
  · rintro rfl
    simp [div_self, fun i => (hq i).ne']

/-- `def-softmax`. -/
noncomputable def softmax (z : ι → ℝ) (i : ι) : ℝ := exp (z i) / ∑ j, exp (z j)

theorem sum_exp_pos [Nonempty ι] (z : ι → ℝ) : 0 < ∑ j, exp (z j) :=
  sum_pos (fun j _ => exp_pos (z j)) univ_nonempty

/-- `def-softmax`: every entry of `softmax z` is positive. -/
theorem softmax_pos [Nonempty ι] (z : ι → ℝ) (i : ι) : 0 < softmax z i :=
  div_pos (exp_pos _) (sum_exp_pos z)

/-- `def-softmax`: the entries of `softmax z` sum to 1. -/
theorem softmax_sum [Nonempty ι] (z : ι → ℝ) : ∑ i, softmax z i = 1 := by
  simp only [softmax, ← sum_div]
  exact div_self (sum_exp_pos z).ne'

/-- `def-softmax`, shift invariance (the algebra behind appendix A's
`thm-softmax-shift`): adding the same constant to every input leaves softmax
unchanged. -/
theorem softmax_add_const (z : ι → ℝ) (c : ℝ) : softmax (fun j => z j + c) = softmax z := by
  funext i
  simp only [softmax, exp_add, ← sum_mul]
  rw [mul_div_mul_right _ _ (exp_pos c).ne']

/-- The indicator `[j = k]`, the `j`th entry of the basis vector `e_k`. -/
def ind [DecidableEq ι] (k j : ι) : ℝ := if j = k then 1 else 0

/-- `thm-softmax-ce-gradient`: the partial derivative of the cross-entropy
loss `L(z) = ln (∑ j exp z j) - z y` with respect to `z k` is
`softmax z k - [k = y]`. The loss is written in its log-sum-exp form,
which equals `-ln (softmax z y)` by `neg_log_softmax`. -/
theorem ce_partial [DecidableEq ι] [Nonempty ι] (z : ι → ℝ) (y k : ι) :
    HasDerivAt
      (fun t : ℝ => log (∑ j, exp (z j + t * ind k j)) - (z y + t * ind k y))
      (softmax z k - ind k y) 0 := by
  have hsum : HasDerivAt (fun t : ℝ => ∑ j, exp (z j + t * ind k j))
      (∑ j, exp (z j + 0 * ind k j) * (1 * ind k j)) 0 :=
    HasDerivAt.fun_sum fun j _ => (((hasDerivAt_id (0 : ℝ)).mul_const (ind k j)).const_add (z j)).exp
  have hlog := hsum.log (by simpa using (sum_exp_pos z).ne')
  have hlin : HasDerivAt (fun t : ℝ => z y + t * ind k y) (1 * ind k y) 0 :=
    ((hasDerivAt_id (0 : ℝ)).mul_const (ind k y)).const_add (z y)
  convert hlog.sub hlin using 1
  simp [softmax, ind]

/-- `thm-softmax-jacobian`: the partial derivative of `softmax z i` with
respect to `z k` is `p i ([i = k] - p k)`, where `p = softmax z`. -/
theorem softmax_partial [DecidableEq ι] [Nonempty ι] (z : ι → ℝ) (i k : ι) :
    HasDerivAt (fun t : ℝ => softmax (fun j => z j + t * ind k j) i)
      (softmax z i * (ind k i - softmax z k)) 0 := by
  have hsum : HasDerivAt (fun t : ℝ => ∑ j, exp (z j + t * ind k j))
      (∑ j, exp (z j + 0 * ind k j) * (1 * ind k j)) 0 :=
    HasDerivAt.fun_sum fun j _ => (((hasDerivAt_id (0 : ℝ)).mul_const (ind k j)).const_add (z j)).exp
  have hnum : HasDerivAt (fun t : ℝ => exp (z i + t * ind k i))
      (exp (z i + 0 * ind k i) * (1 * ind k i)) 0 :=
    (((hasDerivAt_id (0 : ℝ)).mul_const (ind k i)).const_add (z i)).exp
  have hS := (sum_exp_pos z).ne'
  have hdiv := hnum.div hsum (by simpa using hS)
  have hf : (fun t : ℝ => softmax (fun j => z j + t * ind k j) i)
      = (fun t => exp (z i + t * ind k i)) / (fun t => ∑ j, exp (z j + t * ind k j)) := by
    funext t; simp [softmax]
  rw [hf]
  convert hdiv using 1
  simp only [softmax, ind, add_zero, mul_ite, mul_one, mul_zero,
    sum_ite_eq', mem_univ, ite_true, ite_self]
  field_simp
  split_ifs with h
  · subst h; ring
  · ring

/-- `thm-softmax-ce-gradient`, first step of the proof: the cross-entropy
loss `-ln (softmax z y)` equals `ln (∑ j exp z j) - z y`. -/
theorem neg_log_softmax [Nonempty ι] (z : ι → ℝ) (y : ι) :
    -log (softmax z y) = log (∑ j, exp (z j)) - z y := by
  rw [softmax, log_div (exp_pos _).ne' (sum_exp_pos z).ne', log_exp]; ring

/-- `exm-uniform-loss`: if every one of the `|V|` outcomes has
probability `1/|V|`, the cross-entropy loss is `ln |V|`. -/
theorem uniform_loss [Nonempty ι] (z : ι → ℝ) (c : ℝ) (hz : ∀ j, z j = c) (y : ι) :
    -log (softmax z y) = log (Fintype.card ι) := by
  rw [neg_log_softmax]
  simp only [hz, sum_const, card_univ, nsmul_eq_mul]
  rw [log_mul (by exact_mod_cast Fintype.card_ne_zero) (exp_pos c).ne', log_exp]; ring

end AutodiffBook.Ch08
