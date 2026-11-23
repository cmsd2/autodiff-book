import Mathlib

/-!
# Chapter 6: Stochastic and adaptive methods

Machine-checked versions of results the chapter proves by hand, named by
their labels in `chapters/06-stochastic-methods.qmd`.

The exponential moving average is defined by its recurrence, and the
closed form and the bias correction are proved from it. The convergence
theorem for stochastic gradient descent ends with an inequality between
expectations; its final summation step is checked here on sequences of
real numbers, where `F k` stands for `E f(θ_k)` and `a k` for
`E ‖∇f(θ_k)‖²`.
-/

open Finset

namespace AutodiffBook.Ch06

/-- `def-ema`: the exponential moving average `m k` of the values
`g 1, g 2, ...`, with `m 0 = 0` and `m k = β m (k-1) + (1 - β) g k`. -/
def ema (β : ℝ) (g : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | k + 1 => β * ema β g k + (1 - β) * g (k + 1)

/-- `thm-ema-closed`: the closed form
`m k = (1 - β) ∑_{j=1}^{k} β^(k-j) g j`. -/
theorem ema_closed (β : ℝ) (g : ℕ → ℝ) (k : ℕ) :
    ema β g k = (1 - β) * ∑ j ∈ Icc 1 k, β ^ (k - j) * g j := by
  induction k with
  | zero => simp [ema]
  | succ k ih =>
    rw [ema, ih, sum_Icc_succ_top (by omega), mul_add, ← mul_assoc, mul_comm β,
      mul_assoc, mul_sum, mul_sum, mul_sum]
    congr 1
    · apply sum_congr rfl
      intro j hj
      have hj' : j ≤ k := (mem_Icc.mp hj).2
      rw [show k + 1 - j = (k - j) + 1 by omega, pow_succ]
      ring
    · simp

/-- `thm-ema-bias`: if every value equals `c`, then `m k = (1 - β^k) c`. -/
theorem ema_const (β c : ℝ) (k : ℕ) : ema β (fun _ => c) k = (1 - β ^ k) * c := by
  induction k with
  | zero => simp [ema]
  | succ k ih => rw [ema, ih]; ring

/-- `thm-ema-bias`, the weights: the weights `(1 - β) β^(k-j)` of the
closed form sum to `1 - β^k`. -/
theorem ema_weights_sum (β : ℝ) (k : ℕ) :
    (1 - β) * ∑ j ∈ Icc 1 k, β ^ (k - j) = 1 - β ^ k := by
  have h1 := ema_closed β (fun _ => 1) k
  have h2 := ema_const β 1 k
  simp only [mul_one] at h1 h2
  rw [← h1, h2]

/-- `thm-ema-bias`, the correction: for `0 ≤ β < 1` and `k ≥ 1`, dividing
by `1 - β^k` recovers a constant value exactly. -/
theorem ema_bias_corrected (β c : ℝ) (h0 : 0 ≤ β) (h1 : β < 1) (k : ℕ) (hk : 0 < k) :
    ema β (fun _ => c) k / (1 - β ^ k) = c := by
  have : β ^ k < 1 := pow_lt_one₀ h0 h1 hk.ne'
  rw [ema_const]
  field_simp [sub_ne_zero.mpr this.ne']

/-- `thm-sgd-schedule` and `thm-sgd-convergence`, the summation step: if
`F (k+1) ≤ F k - (η k / 2) a k + (C / 2) (η k)^2` for every `k` and
`F` never falls below `fstar`, then
`∑_{k<K} η k a k ≤ 2 (F 0 - fstar) + C ∑_{k<K} (η k)^2`. -/
theorem sgd_sum (F a η : ℕ → ℝ) (C fstar : ℝ)
    (hstep : ∀ k, F (k + 1) ≤ F k - η k / 2 * a k + C / 2 * η k ^ 2)
    (hlow : ∀ k, fstar ≤ F k) (K : ℕ) :
    ∑ k ∈ range K, η k * a k ≤ 2 * (F 0 - fstar) + C * ∑ k ∈ range K, η k ^ 2 := by
  have key : ∀ K, ∑ k ∈ range K, η k * a k ≤ 2 * (F 0 - F K) + C * ∑ k ∈ range K, η k ^ 2 := by
    intro K
    induction K with
    | zero => simp
    | succ K ih =>
      rw [sum_range_succ, sum_range_succ, mul_add]
      linarith [hstep K]
  linarith [key K, hlow K]

/-- `thm-sgd-convergence`: with a constant step `η > 0` and `K ≥ 1`, the
average of the first `K` values of `a` is at most
`2 (F 0 - fstar) / (η K) + η C`. -/
theorem sgd_constant (F a : ℕ → ℝ) (η C fstar : ℝ) (hη : 0 < η)
    (hstep : ∀ k, F (k + 1) ≤ F k - η / 2 * a k + C / 2 * η ^ 2)
    (hlow : ∀ k, fstar ≤ F k) (K : ℕ) (hK : 0 < K) :
    (∑ k ∈ range K, a k) / K ≤ 2 * (F 0 - fstar) / (η * K) + η * C := by
  have h := sgd_sum F a (fun _ => η) C fstar hstep hlow K
  simp only [sum_const, card_range, nsmul_eq_mul, ← mul_sum] at h
  have hK' : (0 : ℝ) < K := by exact_mod_cast hK
  have h' : ∑ k ∈ range K, a k ≤ (2 * (F 0 - fstar) + C * (K * η ^ 2)) / η := by
    rw [le_div_iff₀ hη]; linarith
  calc (∑ k ∈ range K, a k) / K ≤ ((2 * (F 0 - fstar) + C * (K * η ^ 2)) / η) / K :=
        div_le_div_of_nonneg_right h' hK'.le
    _ = 2 * (F 0 - fstar) / (η * K) + η * C := by field_simp

/-- `thm-adam-step`, the inequality of arithmetic and geometric means used
in the proof: `(1 - a)(1 - b) ≤ (1 - √(ab))²` for `a, b ≥ 0`. -/
theorem one_sub_mul_le (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (1 - a) * (1 - b) ≤ (1 - Real.sqrt (a * b)) ^ 2 := by
  have hs : Real.sqrt (a * b) ^ 2 = a * b := Real.sq_sqrt (mul_nonneg ha hb)
  have hamgm : 2 * Real.sqrt (a * b) ≤ a + b := by
    rw [Real.sqrt_mul ha]
    nlinarith [sq_nonneg (Real.sqrt a - Real.sqrt b), Real.sq_sqrt ha, Real.sq_sqrt hb]
  nlinarith

end AutodiffBook.Ch06
