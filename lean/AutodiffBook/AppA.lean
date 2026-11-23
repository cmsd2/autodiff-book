import Mathlib

/-!
# Appendix A: Floating-point arithmetic

Machine-checked versions of results in `chapters/a-floating-point.qmd`,
named by their labels. Vectors in `ℝⁿ` are functions `Fin n → ℝ`.
-/

open Real

namespace AutodiffBook.AppA

/-- The softmax function on `ℝⁿ`. -/
noncomputable def softmax {n : ℕ} (z : Fin n → ℝ) : Fin n → ℝ :=
  fun i => exp (z i) / ∑ j, exp (z j)

/-- `thm-softmax-shift`: adding the same real number `c` to every
component leaves softmax unchanged. -/
theorem softmax_add_const {n : ℕ} (z : Fin n → ℝ) (c : ℝ) :
    softmax (fun j => z j + c) = softmax z := by
  funext i
  simp only [softmax, Real.exp_add, ← Finset.sum_mul]
  exact mul_div_mul_right _ _ (exp_pos c).ne'

/-- `thm-softmax-shift`, the log-sum-exp form: for every real `m`,
`log (∑ exp zⱼ) = m + log (∑ exp (zⱼ - m))`. -/
theorem logSumExp_shift {n : ℕ} (hn : 0 < n) (z : Fin n → ℝ) (m : ℝ) :
    log (∑ j, exp (z j)) = m + log (∑ j, exp (z j - m)) := by
  have hpos : 0 < ∑ j, exp (z j - m) := by
    have : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
    exact Finset.sum_pos (fun j _ => exp_pos _) Finset.univ_nonempty
  have : ∑ j, exp (z j) = exp m * ∑ j, exp (z j - m) := by
    rw [Finset.mul_sum]
    congr 1
    funext j
    rw [← Real.exp_add]
    ring_nf
  rw [this, Real.log_mul (exp_pos m).ne' hpos.ne', Real.log_exp]

/-- `cor-stable-softmax`: after subtracting the maximum, every exponent
is at most `0`, so every term `exp (zⱼ - max z)` lies in `(0, 1]`. -/
theorem exp_sub_max_le_one {n : ℕ} (z : Fin n → ℝ) (m : ℝ) (hm : ∀ j, z j ≤ m) (j : Fin n) :
    0 < exp (z j - m) ∧ exp (z j - m) ≤ 1 :=
  ⟨exp_pos _, Real.exp_le_one_iff.mpr (sub_nonpos.mpr (hm j))⟩

/-- `thm-fd-error`, the combination step: a truncation error of at most
`M₂ h / 2` and two function values each wrong by at most `u M₀` give a
total error of at most `M₂ h / 2 + 2 u M₀ / h`. -/
theorem fd_total_error {f₀ f₁ g₀ g₁ d M₂ M₀ u h : ℝ} (hh : 0 < h)
    (htrunc : |(f₁ - f₀) / h - d| ≤ M₂ * h / 2)
    (h₀ : |g₀ - f₀| ≤ u * M₀) (h₁ : |g₁ - f₁| ≤ u * M₀) :
    |(g₁ - g₀) / h - d| ≤ M₂ * h / 2 + 2 * u * M₀ / h := by
  have hsplit : (g₁ - g₀) / h - d = ((f₁ - f₀) / h - d) + ((g₁ - f₁) - (g₀ - f₀)) / h := by
    field_simp
    ring
  rw [hsplit]
  calc |((f₁ - f₀) / h - d) + ((g₁ - f₁) - (g₀ - f₀)) / h|
      ≤ |(f₁ - f₀) / h - d| + |((g₁ - f₁) - (g₀ - f₀)) / h| := abs_add_le _ _
    _ ≤ M₂ * h / 2 + 2 * u * M₀ / h := by
      gcongr
      rw [abs_div, abs_of_pos hh]
      gcongr
      calc |(g₁ - f₁) - (g₀ - f₀)| ≤ |g₁ - f₁| + |g₀ - f₀| := abs_sub _ _
        _ ≤ u * M₀ + u * M₀ := add_le_add h₁ h₀
        _ = 2 * u * M₀ := by ring

/-- `thm-fd-error`, the best step: for `a, b > 0` and `h > 0`,
`a h + b / h ≥ 2 √(a b)`. -/
theorem step_bound (a b h : ℝ) (ha : 0 < a) (hb : 0 < b) (hh : 0 < h) :
    2 * √(a * b) ≤ a * h + b / h := by
  have hs : √(a * b) ^ 2 = a * b := Real.sq_sqrt (by positivity)
  rw [← sub_nonneg]
  have key : a * h + b / h - 2 * √(a * b) = (a * h - √(a * b)) ^ 2 / (a * h) := by
    field_simp
    nlinarith [hs]
  rw [key]
  positivity

/-- `thm-fd-error`, equality: the bound `2 √(a b)` is attained at
`h = √(b / a)`. -/
theorem step_bound_attained (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    a * √(b / a) + b / √(b / a) = 2 * √(a * b) := by
  have h1 : a * √(b / a) = √(a * b) := by
    rw [show a * b = a ^ 2 * (b / a) by field_simp, Real.sqrt_mul (sq_nonneg a),
      Real.sqrt_sq ha.le]
  have h2 : b / √(b / a) = √(a * b) := by
    have hq : 0 < √(b / a) := Real.sqrt_pos.mpr (div_pos hb ha)
    rw [div_eq_iff hq.ne', ← Real.sqrt_mul (by positivity)]
    rw [show a * b * (b / a) = b ^ 2 by field_simp]
    exact (Real.sqrt_sq hb.le).symm
  rw [h1, h2]
  ring

end AutodiffBook.AppA
