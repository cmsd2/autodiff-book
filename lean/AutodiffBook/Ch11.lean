import Mathlib

/-!
# Chapter 11: A small transformer

Machine-checked versions of results in `chapters/11-small-gpt.qmd`, named
by their labels. Vectors in `ℝⁿ` are functions `Fin n → ℝ`.
-/

open Filter Topology Real

namespace AutodiffBook.Ch11

/-- The (rows, columns) of the nine weight matrices of the model, in the
order of the chapter's table: token embedding, position embedding,
`W_Q`, `W_K`, `W_V`, `W_O`, `W_1`, `W_2` and the output matrix `W_U`. -/
def shapes : List (ℕ × ℕ) :=
  [(27, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (64, 16), (16, 64), (27, 16)]

/-- `thm-parameter-count`: the nine matrices hold 4192 numbers in all. -/
theorem parameter_count : (shapes.map fun s => s.1 * s.2).sum = 4192 := by
  decide

/-- `thm-parameter-count`, general form (exercise `exr-ch11-param-general`):
with vocabulary size `V`, context length `T`, width `d`, `L` layers and an
MLP of width `4d`, the count is `2Vd + Td + 12Ld²`. -/
theorem parameter_count_general (V T d L : ℕ) :
    V * d + T * d + L * (4 * (d * d) + 4 * d * d + d * (4 * d)) + V * d
      = 2 * V * d + T * d + 12 * L * d ^ 2 := by
  ring

/-- `thm-residual-jacobian`: if `F` has derivative `F'` at `a`, then
`x ↦ x + F x` has derivative `I + F'` at `a`. -/
theorem residual_hasFDerivAt {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {F : E → E} {F' : E →L[ℝ] E} {a : E} (hF : HasFDerivAt F F' a) :
    HasFDerivAt (fun x => x + F x) (ContinuousLinearMap.id ℝ E + F') a :=
  (hasFDerivAt_id a).add hF

/-- `def-rmsnorm`: RMS normalisation with parameter `ε` and no gain. -/
noncomputable def rms {n : ℕ} (ε : ℝ) (x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => x i / √((∑ j, x j ^ 2) / n + ε)

/-- `thm-rmsnorm-scaling`: for `c > 0`, normalising `c x` with parameter `ε`
gives the same vector as normalising `x` with parameter `ε / c²`. -/
theorem rms_smul {n : ℕ} (ε c : ℝ) (hc : 0 < c) (x : Fin n → ℝ) :
    rms ε (c • x) = rms (ε / c ^ 2) x := by
  funext i
  simp only [rms, Pi.smul_apply, smul_eq_mul, mul_pow, ← Finset.mul_sum]
  have h : c ^ 2 * (∑ j, x j ^ 2) / n + ε = c ^ 2 * ((∑ j, x j ^ 2) / n + ε / c ^ 2) := by
    field_simp
  rw [h, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq hc.le]
  exact mul_div_mul_left _ _ hc.ne'

/-- `thm-rmsnorm-scaling` with `ε = 0`: normalisation without `ε` is
unchanged by positive scaling. -/
theorem rms_zero_smul {n : ℕ} (c : ℝ) (hc : 0 < c) (x : Fin n → ℝ) :
    rms 0 (c • x) = rms 0 x := by
  rw [rms_smul 0 c hc, zero_div]

/-- The softmax function on `ℝⁿ`. -/
noncomputable def softmax {n : ℕ} (z : Fin n → ℝ) : Fin n → ℝ :=
  fun i => exp (z i) / ∑ j, exp (z j)

/-- `thm-temperature`, high temperature: as `T → ∞`, every component of
`softmax (z / T)` tends to `1 / n`. -/
theorem temperature_atTop {n : ℕ} (z : Fin n → ℝ) (i : Fin n) :
    Tendsto (fun T : ℝ => softmax (fun j => z j / T) i) atTop (𝓝 (1 / n)) := by
  have h1 : ∀ j, Tendsto (fun T : ℝ => exp (z j / T)) atTop (𝓝 1) := fun j => by
    have : Tendsto (fun T : ℝ => z j / T) atTop (𝓝 0) :=
      tendsto_const_nhds.div_atTop tendsto_id
    have h := (continuous_exp.tendsto 0).comp this
    rw [Real.exp_zero] at h
    exact h
  have hs : Tendsto (fun T : ℝ => ∑ j, exp (z j / T)) atTop (𝓝 (∑ _j : Fin n, (1 : ℝ))) :=
    tendsto_finsetSum _ fun j _ => h1 j
  have hn : (∑ _j : Fin n, (1 : ℝ)) ≠ 0 := by
    have : 0 < n := Fin.pos i
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
    positivity
  have := (h1 i).div hs hn
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    mul_one] at this
  exact this

/-- Rewriting softmax with every exponent shifted by the same constant;
the appendix proves this as `thm-softmax-shift`. -/
theorem softmax_shift {n : ℕ} (z : Fin n → ℝ) (c : ℝ) :
    softmax (fun j => z j - c) = softmax z := by
  funext i
  simp only [softmax, Real.exp_sub, ← Finset.sum_div]
  have : (0 : ℝ) < exp c := exp_pos c
  rw [div_div_div_cancel_right₀ this.ne']

/-- `thm-temperature`, low temperature: if `z k` is strictly larger than
every other component, then as `T → 0⁺`, `softmax (z / T)` tends to the
one-hot vector at `k`. -/
theorem temperature_nhdsGT {n : ℕ} (z : Fin n → ℝ) (k : Fin n)
    (hk : ∀ j, j ≠ k → z j < z k) (i : Fin n) :
    Tendsto (fun T : ℝ => softmax (fun j => z j / T) i) (𝓝[>] 0)
      (𝓝 (if i = k then 1 else 0)) := by
  -- Shift every exponent by `z k / T`, so the largest becomes `0`.
  have hshift : ∀ T : ℝ, softmax (fun j => z j / T) i
      = exp ((z i - z k) / T) / ∑ j, exp ((z j - z k) / T) := by
    intro T
    have := congrFun (softmax_shift (fun j => z j / T) (z k / T)) i
    simp only [softmax] at this ⊢
    rw [← this]
    simp only [sub_div]
  have hlim : ∀ j, Tendsto (fun T : ℝ => exp ((z j - z k) / T)) (𝓝[>] 0)
      (𝓝 (if j = k then 1 else 0)) := by
    intro j
    by_cases hj : j = k
    · subst hj
      simp
    · simp only [hj, ite_false]
      have hneg : z j - z k < 0 := sub_neg.mpr (hk j hj)
      have : Tendsto (fun T : ℝ => (z j - z k) * T⁻¹) (𝓝[>] 0) atBot :=
        tendsto_inv_nhdsGT_zero.const_mul_atTop_of_neg hneg
      simp only [div_eq_mul_inv]
      exact Real.tendsto_exp_atBot.comp this
  have hsum : Tendsto (fun T : ℝ => ∑ j, exp ((z j - z k) / T)) (𝓝[>] 0)
      (𝓝 (∑ j, if j = k then (1 : ℝ) else 0)) :=
    tendsto_finsetSum _ fun j _ => hlim j
  rw [Finset.sum_ite_eq' Finset.univ k (fun _ => (1 : ℝ))] at hsum
  simp only [Finset.mem_univ, ite_true] at hsum
  have := (hlim i).div hsum one_ne_zero
  simp only [div_one] at this
  simp only [hshift]
  exact this

end AutodiffBook.Ch11
