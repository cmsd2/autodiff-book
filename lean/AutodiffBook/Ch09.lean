import Mathlib

/-!
# Chapter 9: Language models

Machine-checked versions of results that chapter 9 proves by hand, named
by their labels in `chapters/09-language-models.qmd`. A vocabulary is any
finite type `V`; a row of a bigram model is a function `V → ℝ`.
-/

open Finset Real

namespace AutodiffBook.Ch09

variable {V : Type*} [Fintype V]

/-- `thm-bigram-mle`, normalisation: in a row with at least one observed
pair, the relative frequencies `n a b / n a` sum to 1, so they form a
probability distribution over the next token. -/
theorem mle_row_sum (n : V → V → ℕ) (a : V) (h : 0 < ∑ b, n a b) :
    ∑ b, (n a b : ℝ) / ∑ c, (n a c : ℝ) = 1 := by
  rw [← Finset.sum_div]
  have hpos : (0 : ℝ) < ∑ c, (n a c : ℝ) := by exact_mod_cast h
  exact div_self hpos.ne'

/-- The inequality behind `lem-gibbs-counts`, one term at a time: for a
count `c ≥ 0`, a total `N > 0` and a probability `p > 0`,
`c ln p - c ln (c/N) ≤ N p - c`. -/
theorem term_bound (c N p : ℝ) (hc : 0 ≤ c) (hN : 0 < N) (hp : 0 < p) :
    c * log p - c * log (c / N) ≤ N * p - c := by
  rcases hc.eq_or_lt with h | h
  · subst h; simp; positivity
  · have hq : 0 < p * N / c := by positivity
    have key := log_le_sub_one_of_pos hq
    have hlog : log (p * N / c) = log p - log (c / N) := by
      rw [show p * N / c = p / (c / N) by field_simp]
      exact log_div hp.ne' (by positivity)
    rw [hlog] at key
    have := mul_le_mul_of_nonneg_left key h.le
    have hsimp : c * (p * N / c - 1) = N * p - c := by field_simp
    linarith [this, hsimp]

/-- `thm-bigram-mle`, the maximum: in one row of a bigram model, with
counts `c b ≥ 0` of total `N > 0`, every distribution `p` with positive
entries has log-likelihood `∑ c b ln p b` at most that of the relative
frequencies `c b / N`. -/
theorem bigram_mle_row (c p : V → ℝ) (hc : ∀ b, 0 ≤ c b) (hN : 0 < ∑ b, c b)
    (hp : ∀ b, 0 < p b) (hsum : ∑ b, p b = 1) :
    ∑ b, c b * log (p b) ≤ ∑ b, c b * log (c b / ∑ b', c b') := by
  have h := Finset.sum_le_sum fun b (_ : b ∈ (univ : Finset V)) =>
    term_bound (c b) (∑ b', c b') (p b) (hc b) hN (hp b)
  rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hsum] at h
  linarith

/-- `thm-uniform-perplexity`: if every one of `N > 0`
predictions has probability `1/k`, the average negative log-likelihood is
`ln k`. -/
theorem uniform_loss (N k : ℕ) (hN : 0 < N) :
    -(1 / (N : ℝ)) * ∑ _i : Fin N, log (1 / (k : ℝ)) = log k := by
  have hN' : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, one_div,
    log_inv]
  field_simp

/-- `thm-uniform-perplexity`: the uniform model's perplexity `exp (ln k)` is `k`. -/
theorem uniform_perplexity (k : ℕ) (hk : 0 < k) : exp (log (k : ℝ)) = k :=
  exp_log (by exact_mod_cast hk)

/-- `thm-neural-bigram`: taking the logits to be the logarithms of a
distribution with positive entries, softmax returns that distribution. -/
theorem softmax_log (p : V → ℝ) (hp : ∀ b, 0 < p b) (hsum : ∑ b, p b = 1) (b : V) :
    exp (log (p b)) / ∑ c, exp (log (p c)) = p b := by
  have h : ∑ c, exp (log (p c)) = ∑ c, p c :=
    Finset.sum_congr rfl fun c _ => exp_log (hp c)
  rw [h, hsum, exp_log (hp b), div_one]

end AutodiffBook.Ch09
