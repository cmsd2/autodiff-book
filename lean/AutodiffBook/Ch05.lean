import Mathlib

/-!
# Chapter 5: Gradient descent

Machine-checked versions of results the chapter proves by hand, named by
their labels in `chapters/05-gradient-descent.qmd`.

The convergence theorems for gradient descent combine an analytic step
(the descent lemma, or convexity) with an algebraic step on sequences of
real numbers. The algebraic steps are checked here as statements about
sequences: `F k` stands for `f(θ_k)`, `a k` for `‖∇f(θ_k)‖²`, and
`fstar` for the infimum of `f`. The first-order characterisation of
convexity is checked in any real inner product space, and the step-size
analysis for quadratics is checked one eigenvalue at a time.
-/

open scoped InnerProductSpace
open Finset

namespace AutodiffBook.Ch05

/-- `thm-gd-nonconvex`, the telescoping step: if every step lowers `F` by
at least `(η/2) a k` and `F` never falls below `fstar`, then the first `k`
values of `a` sum to at most `2 (F 0 - fstar) / η`. -/
theorem gd_nonconvex_sum (F a : ℕ → ℝ) (η fstar : ℝ) (hη : 0 < η)
    (hstep : ∀ j, F (j + 1) ≤ F j - η / 2 * a j) (hlow : ∀ j, fstar ≤ F j)
    (k : ℕ) : ∑ j ∈ range k, a j ≤ 2 * (F 0 - fstar) / η := by
  have key : ∀ k, η / 2 * ∑ j ∈ range k, a j ≤ F 0 - F k := by
    intro k
    induction k with
    | zero => simp
    | succ k ih =>
      rw [sum_range_succ, mul_add]
      linarith [hstep k]
  rw [le_div_iff₀ hη]
  linarith [key k, hlow k]

/-- `thm-gd-nonconvex`, the minimum: the smallest of the first `k ≥ 1`
values of `a` is at most `2 (F 0 - fstar) / (η k)`. -/
theorem gd_nonconvex_min (F a : ℕ → ℝ) (η fstar : ℝ) (hη : 0 < η)
    (hstep : ∀ j, F (j + 1) ≤ F j - η / 2 * a j) (hlow : ∀ j, fstar ≤ F j)
    (k : ℕ) (hk : 0 < k) :
    ∃ j < k, a j ≤ 2 * (F 0 - fstar) / (η * k) := by
  by_contra! h
  have hsum := gd_nonconvex_sum F a η fstar hη hstep hlow k
  have hlt : ∑ _j ∈ range k, 2 * (F 0 - fstar) / (η * k) < ∑ j ∈ range k, a j :=
    sum_lt_sum_of_nonempty (nonempty_range_iff.mpr hk.ne') (fun j hj => h j (mem_range.mp hj))
  rw [sum_const, card_range, nsmul_eq_mul] at hlt
  have hk' : (0 : ℝ) < k := by exact_mod_cast hk
  have : (k : ℝ) * (2 * (F 0 - fstar) / (η * k)) = 2 * (F 0 - fstar) / η := by
    field_simp
  linarith

/-- `thm-gd-strongly-convex`, the induction: if each step multiplies the
gap `F k - fstar` by at most `1 - c`, with `0 ≤ 1 - c`, then after `k` steps
the gap is at most `(1 - c)^k` times the first gap. -/
theorem linear_rate (F : ℕ → ℝ) (c fstar : ℝ) (hc : 0 ≤ 1 - c)
    (hstep : ∀ k, F (k + 1) - fstar ≤ (1 - c) * (F k - fstar)) (k : ℕ) :
    F k - fstar ≤ (1 - c) ^ k * (F 0 - fstar) := by
  induction k with
  | zero => simp
  | succ k ih =>
    calc F (k + 1) - fstar ≤ (1 - c) * (F k - fstar) := hstep k
      _ ≤ (1 - c) * ((1 - c) ^ k * (F 0 - fstar)) := mul_le_mul_of_nonneg_left ih hc
      _ = (1 - c) ^ (k + 1) * (F 0 - fstar) := by ring

/-- `thm-gd-convex`, the summation: if `F (j+1) - fstar ≤ (D j - D (j+1)) / (2η)`
for every `j` (with `D j = ‖θ_j - θ*‖²`), `D` is non-negative and `F` is
non-increasing, then `F k - fstar ≤ D 0 / (2 η k)` for `k ≥ 1`. -/
theorem gd_convex_rate (F D : ℕ → ℝ) (η fstar : ℝ) (hη : 0 < η)
    (hstep : ∀ j, F (j + 1) - fstar ≤ (D j - D (j + 1)) / (2 * η))
    (hD : ∀ j, 0 ≤ D j) (hmono : ∀ j, F (j + 1) ≤ F j) (k : ℕ) (hk : 0 < k) :
    F k - fstar ≤ D 0 / (2 * η * k) := by
  have hanti : ∀ i j, i ≤ j → F j ≤ F i := by
    intro i j hij
    induction j, hij using Nat.le_induction with
    | base => exact le_rfl
    | succ j _ ih => exact (hmono j).trans ih
  have tele : ∀ m, ∑ j ∈ range m, (F (j + 1) - fstar) ≤ (D 0 - D m) / (2 * η) := by
    intro m
    induction m with
    | zero => simp
    | succ m ih =>
      rw [sum_range_succ]
      have := hstep m
      calc _ ≤ (D 0 - D m) / (2 * η) + (D m - D (m + 1)) / (2 * η) := add_le_add ih this
        _ = (D 0 - D (m + 1)) / (2 * η) := by field_simp; ring
  have lower : ∑ _j ∈ range k, (F k - fstar) ≤ ∑ j ∈ range k, (F (j + 1) - fstar) :=
    sum_le_sum fun j hj => by linarith [hanti (j + 1) k (mem_range.mp hj)]
  rw [sum_const, card_range, nsmul_eq_mul] at lower
  have hk' : (0 : ℝ) < k := by exact_mod_cast hk
  have h2 : (D 0 - D k) / (2 * η) ≤ D 0 / (2 * η) := by
    apply div_le_div_of_nonneg_right _ (by positivity); linarith [hD k]
  have h3 : (k : ℝ) * (F k - fstar) ≤ D 0 / (2 * η) := by linarith [tele k]
  rw [le_div_iff₀ (by positivity)] at h3
  rw [le_div_iff₀ (by positivity)]
  calc (F k - fstar) * (2 * η * k) = k * (F k - fstar) * (2 * η) := by ring
    _ ≤ D 0 := h3

/-- `thm-convex-first-order`, the converse direction: a function that lies
above all its tangent planes is convex. Here `g x` plays the role of
`∇f(x)`; the statement holds in any real inner product space. -/
theorem convex_of_tangent {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (f : E → ℝ) (g : E → E) (h : ∀ x y, f x + ⟪g x, y - x⟫_ℝ ≤ f y) :
    ConvexOn ℝ Set.univ f := by
  refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
  set z := a • x + b • y
  have hx := h z x
  have hy := h z y
  have hsum : a • (x - z) + b • (y - z) = 0 := by
    have e : a • (x - z) + b • (y - z) = (a • x + b • y) - (a + b) • z := by
      simp only [smul_sub, add_smul]; abel
    rw [e, hab, one_smul]
    exact sub_self _
  have hinner : a * ⟪g z, x - z⟫_ℝ + b * ⟪g z, y - z⟫_ℝ = 0 := by
    rw [← real_inner_smul_right, ← real_inner_smul_right, ← inner_add_right, hsum,
      inner_zero_right]
  have e1 := mul_le_mul_of_nonneg_left hx ha
  have e2 := mul_le_mul_of_nonneg_left hy hb
  rw [mul_add] at e1 e2
  have hz : a * f z + b * f z = f z := by rw [← add_mul, hab, one_mul]
  simp only [smul_eq_mul]
  linarith

/-- `thm-gd-quadratic`, one eigenvalue: for `λ > 0` the factor `1 - ηλ`
has absolute value less than 1 exactly when `0 < η < 2/λ`. -/
theorem contraction_iff (η lam : ℝ) (hl : 0 < lam) :
    |1 - η * lam| < 1 ↔ 0 < η ∧ η < 2 / lam := by
  rw [abs_lt, lt_div_iff₀ hl]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨by nlinarith, by linarith⟩
  · rintro ⟨h1, h2⟩
    exact ⟨by linarith, by nlinarith⟩

/-- `thm-gd-quadratic`, the error of one eigencomponent: iterating
`e ↦ (1 - ηλ) e` from `e₀` gives `(1 - ηλ)^k e₀`, which tends to zero
when `0 < η < 2/λ`. -/
theorem component_tendsto_zero (η lam e₀ : ℝ) (hl : 0 < lam) (h1 : 0 < η)
    (h2 : η < 2 / lam) :
    Filter.Tendsto (fun k : ℕ => (1 - η * lam) ^ k * e₀) Filter.atTop (nhds 0) := by
  have h := (contraction_iff η lam hl).mpr ⟨h1, h2⟩
  simpa using (tendsto_pow_atTop_nhds_zero_of_abs_lt_one h).mul_const e₀

/-- `thm-gd-quadratic`, the optimal step: for `0 < μ ≤ L`, no step size `η`
makes the worse of the two factors `|1 - ημ|` and `|1 - ηL|` smaller than
`(L - μ)/(L + μ)`. -/
theorem optimal_rate_le (μ L η : ℝ) (hμ : 0 < μ) (hμL : μ ≤ L) :
    (L - μ) / (L + μ) ≤ max |1 - η * μ| |1 - η * L| := by
  rw [div_le_iff₀ (by linarith)]
  rcases le_total η (2 / (μ + L)) with h | h
  · have h' : η * (μ + L) ≤ 2 := by rwa [le_div_iff₀ (by linarith)] at h
    have : (L - μ) ≤ (1 - η * μ) * (L + μ) := by nlinarith
    calc L - μ ≤ (1 - η * μ) * (L + μ) := this
      _ ≤ |1 - η * μ| * (L + μ) :=
        mul_le_mul_of_nonneg_right (le_abs_self _) (by linarith)
      _ ≤ max |1 - η * μ| |1 - η * L| * (L + μ) :=
        mul_le_mul_of_nonneg_right (le_max_left _ _) (by linarith)
  · have h' : 2 ≤ η * (μ + L) := by rwa [div_le_iff₀ (by linarith)] at h
    have : (L - μ) ≤ (η * L - 1) * (L + μ) := by nlinarith
    calc L - μ ≤ (η * L - 1) * (L + μ) := this
      _ ≤ |1 - η * L| * (L + μ) :=
        mul_le_mul_of_nonneg_right (by rw [abs_sub_comm]; exact le_abs_self _) (by linarith)
      _ ≤ max |1 - η * μ| |1 - η * L| * (L + μ) :=
        mul_le_mul_of_nonneg_right (le_max_right _ _) (by linarith)

/-- `thm-gd-quadratic`, the optimal step attains the bound: at
`η = 2/(μ + L)` both factors equal `(L - μ)/(L + μ)` in absolute value. -/
theorem optimal_rate_eq (μ L : ℝ) (hμ : 0 < μ) (hμL : μ ≤ L) :
    max |1 - 2 / (μ + L) * μ| |1 - 2 / (μ + L) * L| = (L - μ) / (L + μ) := by
  have hpos : 0 < μ + L := by linarith
  have hne : μ + L ≠ 0 := hpos.ne'
  have hne' : L + μ ≠ 0 := by linarith
  have e1 : 1 - 2 / (μ + L) * μ = (L - μ) / (L + μ) := by field_simp; ring
  have e2 : 1 - 2 / (μ + L) * L = -((L - μ) / (L + μ)) := by field_simp; ring
  have hnn : 0 ≤ (L - μ) / (L + μ) := div_nonneg (by linarith) (by linarith)
  rw [e1, e2, abs_neg, abs_of_nonneg hnn, max_self]

end AutodiffBook.Ch05
