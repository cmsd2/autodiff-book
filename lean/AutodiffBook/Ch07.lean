import Mathlib

/-!
# Chapter 7: Neural networks

Machine-checked versions of identities that chapter 7
(`chapters/07-networks.qmd`) uses about the ReLU function
`relu x = max x 0`.
-/

open Finset

namespace AutodiffBook.Ch07

/-- The rectified linear unit, `def-relu`. -/
noncomputable def relu (x : ℝ) : ℝ := max x 0

/-- `def-relu`: `relu x = x` for `x ≥ 0`. -/
theorem relu_of_nonneg {x : ℝ} (hx : 0 ≤ x) : relu x = x := max_eq_left hx

/-- `def-relu`: `relu x = 0` for `x ≤ 0`. -/
theorem relu_of_nonpos {x : ℝ} (hx : x ≤ 0) : relu x = 0 := max_eq_right hx

/-- `exm-relu-ad-wrong`: `relu x - relu (-x) = x` for every `x`, so this
difference of ReLUs is the identity function, with derivative 1 at 0. -/
theorem relu_sub_relu_neg (x : ℝ) : relu x - relu (-x) = x := by
  rcases le_total 0 x with hx | hx
  · rw [relu_of_nonneg hx, relu_of_nonpos (by linarith)]; ring
  · rw [relu_of_nonpos hx, relu_of_nonneg (by linarith)]; ring

/-- `lem-relu-second-moment`: `relu x ^ 2 + relu (-x) ^ 2 = x ^ 2`, the identity
behind `E[relu(z)²] = Var(z)/2` for a symmetric `z`. -/
theorem relu_sq_add_relu_neg_sq (x : ℝ) : relu x ^ 2 + relu (-x) ^ 2 = x ^ 2 := by
  rcases le_total 0 x with hx | hx
  · rw [relu_of_nonneg hx, relu_of_nonpos (by linarith)]; ring
  · rw [relu_of_nonpos hx, relu_of_nonneg (by linarith)]; ring

/-- `lem-relu-sum`, the computation in its proof: on the piece
`[t j, t (j+1)]` between consecutive breakpoints, the ReLU sum
`c + ∑_{k<N} d k * relu (x - t k)` is the affine function
`c + ∑_{k≤j} d k * (x - t k)`, because the units with `k ≤ j` are active
and the rest are off. -/
theorem relu_sum_on_piece (t d : ℕ → ℝ) (ht : Monotone t) (c x : ℝ) {N j : ℕ}
    (hjN : j < N) (hx₁ : t j ≤ x) (hx₂ : x ≤ t (j + 1)) :
    c + ∑ k ∈ range N, d k * relu (x - t k) = c + ∑ k ∈ range (j + 1), d k * (x - t k) := by
  rw [← sum_range_add_sum_Ico _ (Nat.succ_le_of_lt hjN)]
  have hon : ∑ k ∈ range (j + 1), d k * relu (x - t k)
      = ∑ k ∈ range (j + 1), d k * (x - t k) := by
    refine sum_congr rfl fun k hk => ?_
    have hk : k ≤ j := Nat.lt_succ_iff.mp (mem_range.mp hk)
    rw [relu_of_nonneg (by linarith [ht hk])]
  have hoff : ∑ k ∈ Ico (j + 1) N, d k * relu (x - t k) = 0 := by
    refine sum_eq_zero fun k hk => ?_
    have hk : j + 1 ≤ k := (mem_Ico.mp hk).1
    rw [relu_of_nonpos (by linarith [ht hk]), mul_zero]
  rw [hon, hoff, add_zero]

/-- `lem-relu-sum`, the slopes: if `d 0 = m 0` and `d (k+1) = m (k+1) - m k`,
the slope `∑_{k≤j} d k` of the ReLU sum on piece `j` is `m j`. -/
theorem slope_telescopes (m d : ℕ → ℝ) (h0 : d 0 = m 0)
    (hd : ∀ k, d (k + 1) = m (k + 1) - m k) (j : ℕ) :
    ∑ k ∈ range (j + 1), d k = m j := by
  induction j with
  | zero => simp [h0]
  | succ j ih => rw [sum_range_succ, ih, hd]; ring

/-- `exm-relu-square`: the network `x/N + (2/N) ∑_{k=1}^{N-1} relu (x - k/N)`
agrees with `x ^ 2` at every grid point `j/N` with `j ≤ N`. -/
theorem square_grid (N j : ℕ) (hN : 0 < N) (hj : j ≤ N) :
    ((j : ℝ) / N) / N + (2 / N) * ∑ k ∈ Ico 1 N, relu ((j : ℝ) / N - k / N)
      = ((j : ℝ) / N) ^ 2 := by
  have hN' : (0 : ℝ) < N := by exact_mod_cast hN
  -- Split the sum at `j`: the units with `k < j` are active, the rest are off.
  have key : ∀ i ≤ N, ∑ k ∈ Ico 1 N, relu ((i : ℝ) / N - k / N)
      = ((i : ℝ) * (i - 1) / 2) / N := by
    intro i hi
    induction i with
    | zero =>
      simp only [CharP.cast_eq_zero, zero_div, zero_sub, zero_mul]
      refine sum_eq_zero fun k _ => relu_of_nonpos ?_
      have : (0 : ℝ) ≤ k / N := by positivity
      linarith
    | succ i ih =>
      have hiN : i < N := hi
      have ih := ih hiN.le
      have step : ∀ k ∈ Ico 1 N, relu (((i + 1 : ℕ) : ℝ) / N - k / N)
          = relu ((i : ℝ) / N - k / N) + (if k ≤ i then (1 : ℝ) / N else 0) := by
        intro k _
        push_cast
        split_ifs with hk
        · have hk' : (k : ℝ) ≤ i := by exact_mod_cast hk
          rw [relu_of_nonneg, relu_of_nonneg] <;>
            [ring; (rw [← sub_div]; apply div_nonneg <;> linarith);
              (rw [← sub_div]; apply div_nonneg <;> linarith)]
        · have hk' : (i : ℝ) + 1 ≤ k := by exact_mod_cast Nat.lt_of_not_le hk
          rw [relu_of_nonpos, relu_of_nonpos] <;>
            [ring; (rw [← sub_div]; apply div_nonpos_of_nonpos_of_nonneg <;> linarith);
              (rw [← sub_div]; apply div_nonpos_of_nonpos_of_nonneg <;> linarith)]
      rw [sum_congr rfl step, sum_add_distrib, ih, ← sum_filter]
      have hcount : (Ico 1 N).filter (fun k => k ≤ i) = Ico 1 (i + 1) := by
        ext k; simp only [mem_filter, mem_Ico]; omega
      rw [hcount, sum_const, Nat.card_Ico, nsmul_eq_mul]
      push_cast
      field_simp
      ring
  rw [key j hj]
  field_simp
  ring

end AutodiffBook.Ch07
