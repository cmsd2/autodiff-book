import Mathlib

/-!
# Chapter 10: Attention

Machine-checked versions of results that chapter 10 proves by hand, named
by their labels in `chapters/10-attention.qmd`. Positions are `Fin T`,
vectors are functions `Fin n → ℝ`, and `⬝ᵥ` is the dot product.
-/

open Finset Real Matrix

namespace AutodiffBook.Ch10

variable {T dh dv : ℕ}

/-- The attention score of a query against a key, scaled by `1/√d_h`. -/
noncomputable def score (q k : Fin dh → ℝ) : ℝ := q ⬝ᵥ k / √(dh : ℝ)

/-- `def-attention` for one query: the softmax-weighted sum of the values. -/
noncomputable def attend {ι : Type*} [Fintype ι] (q : Fin dh → ℝ) (k : ι → Fin dh → ℝ)
    (v : ι → Fin dv → ℝ) : Fin dv → ℝ :=
  ∑ s, (exp (score q (k s)) / ∑ r, exp (score q (k r))) • v s

/-- Self-attention of a sequence `X` (rows `X t`) with query, key and value
matrices. -/
noncomputable def selfAttention {d : ℕ} (WQ WK : Matrix (Fin dh) (Fin d) ℝ)
    (WV : Matrix (Fin dv) (Fin d) ℝ) (X : Fin T → Fin d → ℝ) (t : Fin T) : Fin dv → ℝ :=
  attend (WQ *ᵥ X t) (fun s => WK *ᵥ X s) (fun s => WV *ᵥ X s)

/-- `cor-causal-multiset` and the heart of `thm-permutation-equivariance`:
attention for one query does not depend on the order in which the
key–value pairs are listed. -/
theorem attend_perm {ι : Type*} [Fintype ι] (σ : Equiv.Perm ι) (q : Fin dh → ℝ)
    (k : ι → Fin dh → ℝ) (v : ι → Fin dv → ℝ) :
    attend q (k ∘ σ) (v ∘ σ) = attend q k v := by
  unfold attend
  have hden : ∑ r, exp (score q ((k ∘ σ) r)) = ∑ r, exp (score q (k r)) :=
    Equiv.sum_comp σ (fun r => exp (score q (k r)))
  rw [hden]
  exact Equiv.sum_comp σ (fun s => (exp (score q (k s)) / ∑ r, exp (score q (k r))) • v s)

/-- `thm-permutation-equivariance`: permuting the positions of the input
permutes the outputs of (unmasked) self-attention in the same way. -/
theorem permutation_equivariance {d : ℕ} (WQ WK : Matrix (Fin dh) (Fin d) ℝ)
    (WV : Matrix (Fin dv) (Fin d) ℝ) (X : Fin T → Fin d → ℝ) (σ : Equiv.Perm (Fin T))
    (t : Fin T) :
    selfAttention WQ WK WV (X ∘ σ) t = selfAttention WQ WK WV X (σ t) := by
  unfold selfAttention
  exact attend_perm σ (WQ *ᵥ X (σ t)) (fun s => WK *ᵥ X s) (fun s => WV *ᵥ X s)

/-- `def-causal-mask`: masked attention at position `t`, in which only the
positions `s ≤ t` receive weight. -/
noncomputable def causalAttend (q : Fin dh → ℝ) (k : Fin T → Fin dh → ℝ)
    (v : Fin T → Fin dv → ℝ) (t : Fin T) : Fin dv → ℝ :=
  ∑ s ∈ univ.filter (· ≤ t),
    (exp (score q (k s)) / ∑ r ∈ univ.filter (· ≤ t), exp (score q (k r))) • v s

/-- `thm-causal-cache`, first half: the output of masked attention at
position `t` does not depend on the keys and values at later positions. -/
theorem causal_local (q : Fin dh → ℝ) (k k' : Fin T → Fin dh → ℝ)
    (v v' : Fin T → Fin dv → ℝ) (t : Fin T)
    (hk : ∀ s ≤ t, k s = k' s) (hv : ∀ s ≤ t, v s = v' s) :
    causalAttend q k v t = causalAttend q k' v' t := by
  unfold causalAttend
  have hden : ∑ r ∈ univ.filter (· ≤ t), exp (score q (k r))
      = ∑ r ∈ univ.filter (· ≤ t), exp (score q (k' r)) :=
    Finset.sum_congr rfl fun r hr => by rw [hk r (by simpa using hr)]
  rw [hden]
  refine Finset.sum_congr rfl fun s hs => ?_
  have hs' : s ≤ t := by simpa using hs
  rw [hk s hs', hv s hs']

/-- `thm-causal-cache`, second half: masked attention at position `t` equals
unmasked attention over the first `t + 1` positions only, the computation
that a key–value cache performs. -/
theorem causal_eq_prefix (q : Fin dh → ℝ) (k : Fin T → Fin dh → ℝ)
    (v : Fin T → Fin dv → ℝ) (t : Fin T) :
    causalAttend q k v t =
      attend q (fun s : Fin (t.val + 1) => k (Fin.castLE t.isLt s))
        (fun s : Fin (t.val + 1) => v (Fin.castLE t.isLt s)) := by
  unfold causalAttend attend
  -- Both sums run over the positions `0, …, t`, listed in two ways.
  have reindex : ∀ f : Fin T → (Fin dv → ℝ),
      ∑ s ∈ univ.filter (· ≤ t), f s = ∑ s : Fin (t.val + 1), f (Fin.castLE t.isLt s) := by
    intro f
    refine Finset.sum_bij' (fun s hs => ⟨s.val, by
        have : s ≤ t := by simpa using hs
        exact Nat.lt_succ_of_le this⟩)
      (fun s _ => Fin.castLE t.isLt s) ?_ ?_ ?_ ?_ ?_
    · intro s _; simp
    · intro s _
      simp only [mem_filter, mem_univ, true_and]
      exact Fin.le_iff_val_le_val.mpr (Nat.lt_succ_iff.mp s.isLt)
    · intro s _; rfl
    · intro s _; rfl
    · intro s _; rfl
  have reindexR : ∀ f : Fin T → ℝ,
      ∑ s ∈ univ.filter (· ≤ t), f s = ∑ s : Fin (t.val + 1), f (Fin.castLE t.isLt s) := by
    intro f
    refine Finset.sum_bij' (fun s hs => ⟨s.val, by
        have : s ≤ t := by simpa using hs
        exact Nat.lt_succ_of_le this⟩)
      (fun s _ => Fin.castLE t.isLt s) ?_ ?_ ?_ ?_ ?_
    · intro s _; simp
    · intro s _
      simp only [mem_filter, mem_univ, true_and]
      exact Fin.le_iff_val_le_val.mpr (Nat.lt_succ_iff.mp s.isLt)
    · intro s _; rfl
    · intro s _; rfl
    · intro s _; rfl
  rw [reindexR (fun r => exp (score q (k r)))]
  exact reindex (fun s => (exp (score q (k s)) /
    ∑ r : Fin (t.val + 1), exp (score q (k (Fin.castLE t.isLt r)))) • v s)

section Scale

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

omit [IsProbabilityMeasure μ] in
/-- For independent `X`, `Y` with mean 0 and mean square 1, the product has
mean 0 and mean square 1, and lies in `L²`. -/
theorem product_moments {X Y : Ω → ℝ} (hXY : IndepFun X Y μ)
    (hXm : Measurable X) (hYm : Measurable Y)
    (hX : MemLp X 2 μ) (hY : MemLp Y 2 μ)
    (hX0 : μ[X] = 0) (hX1 : μ[X ^ 2] = 1) (hY1 : μ[Y ^ 2] = 1) :
    μ[X * Y] = 0 ∧ μ[(X * Y) ^ 2] = 1 ∧ MemLp (X * Y) 2 μ := by
  have hsq : IndepFun (X ^ 2) (Y ^ 2) μ := by
    have := hXY.comp (φ := fun x : ℝ => x ^ 2) (ψ := fun x : ℝ => x ^ 2)
      (by fun_prop) (by fun_prop)
    simpa [Function.comp_def, Pi.pow_def] using this
  have hmul2 : (X * Y) ^ 2 = X ^ 2 * Y ^ 2 := by
    funext ω; simp [Pi.mul_apply, Pi.pow_apply, mul_pow]
  refine ⟨?_, ?_, ?_⟩
  · rw [hXY.integral_mul_eq_mul_integral hXm.aestronglyMeasurable hYm.aestronglyMeasurable,
      hX0, zero_mul]
  · rw [hmul2, hsq.integral_mul_eq_mul_integral (by fun_prop) (by fun_prop), hX1, hY1, one_mul]
  · rw [memLp_two_iff_integrable_sq (by fun_prop)]
    have hint : Integrable (X ^ 2 * Y ^ 2) μ :=
      hsq.integrable_mul (by simpa [Pi.pow_def] using hX.integrable_sq)
        (by simpa [Pi.pow_def] using hY.integrable_sq)
    have : (fun ω => (X * Y) ω ^ 2) = X ^ 2 * Y ^ 2 := by
      funext ω; simp [mul_pow]
    rw [this]; exact hint

/-- `thm-attention-scale`, in the case the chapter singles out, where all
`2 d_h` components of `q` and `k` are independent with mean 0 and variance 1:
then `q ⬝ k` has mean 0 and variance `d_h`. The components are `Z (inl i)`
for `q` and `Z (inr i)` for `k`, on any probability space (not only a
finite one). -/
theorem attention_scale {dh : ℕ} (Z : Fin dh ⊕ Fin dh → Ω → ℝ) (hind : iIndepFun Z μ)
    (hmeas : ∀ i, Measurable (Z i)) (hL2 : ∀ i, MemLp (Z i) 2 μ)
    (h0 : ∀ i, μ[Z i] = 0) (h1 : ∀ i, μ[Z i ^ 2] = 1) :
    μ[∑ i, Z (.inl i) * Z (.inr i)] = 0 ∧
      variance (∑ i, Z (.inl i) * Z (.inr i)) μ = dh := by
  have hpair : ∀ i, IndepFun (Z (.inl i)) (Z (.inr i)) μ := fun i =>
    hind.indepFun (by simp)
  have hm := fun i => product_moments (hpair i) (hmeas _) (hmeas _) (hL2 _) (hL2 _)
    (h0 _) (h1 _) (h1 _)
  have hint : ∀ i, Integrable (Z (.inl i) * Z (.inr i)) μ := fun i =>
    (hm i).2.2.integrable (by norm_num)
  constructor
  · simp only [Finset.sum_apply]
    rw [integral_finsetSum _ (fun i _ => hint i)]
    exact Finset.sum_eq_zero fun i _ => (hm i).1
  · rw [IndepFun.variance_sum (fun i _ => (hm i).2.2)]
    · have : ∀ i, variance (Z (.inl i) * Z (.inr i)) μ = 1 := by
        intro i
        rw [variance_of_integral_eq_zero (by fun_prop) (hm i).1]
        simpa [Pi.pow_apply] using (hm i).2.1
      simp [this]
    · intro i _ j _ hij
      exact hind.indepFun_mul_mul hmeas _ _ _ _ (fun h => hij (Sum.inl_injective h))
        Sum.inl_ne_inr Sum.inr_ne_inl (fun h => hij (Sum.inr_injective h))

end Scale

end AutodiffBook.Ch10
