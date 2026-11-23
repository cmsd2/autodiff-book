import Mathlib

/-!
# Chapter 2: The chain rule

Machine-checked versions of results the chapter proves by hand, named by
their labels in `chapters/02-chain-rule.qmd`. As in chapter 1, statements
about derivatives use `EuclideanSpace ℝ (Fin n)`, which is ℝⁿ with the dot
product and the Euclidean norm. `HasFDerivAt f L a` is the book's
`def-differentiable` with `Df(a) = L`.
-/

open scoped InnerProductSpace Matrix

namespace AutodiffBook.Ch02

variable {n m p : ℕ}

local notation "ℝ^" k => EuclideanSpace ℝ (Fin k)

/-- `thm-chain-rule`: if `f` is differentiable at `a` and `g` at `f a`, then
`g ∘ f` is differentiable at `a` with `D(g ∘ f)(a) = Dg(f(a)) Df(a)`. -/
theorem chain_rule {f : (ℝ^n) → ℝ^m} {g : (ℝ^m) → ℝ^p} {a : ℝ^n}
    {Df : (ℝ^n) →L[ℝ] ℝ^m} {Dg : (ℝ^m) →L[ℝ] ℝ^p}
    (hf : HasFDerivAt f Df a) (hg : HasFDerivAt g Dg (f a)) :
    HasFDerivAt (g ∘ f) (Dg.comp Df) a :=
  hg.comp a hf

/-- `thm-chain-rule`, matrix form: the matrix of a composition of linear maps
(in the standard bases) is the product of their matrices, so
`J_{g∘f} = J_g J_f`. -/
theorem jacobian_comp (Dg : (ℝ^m) →L[ℝ] ℝ^p) (Df : (ℝ^n) →L[ℝ] ℝ^m) :
    LinearMap.toMatrix (EuclideanSpace.basisFun (Fin n) ℝ).toBasis
        (EuclideanSpace.basisFun (Fin p) ℝ).toBasis (Dg.comp Df : (ℝ^n) →ₗ[ℝ] ℝ^p) =
      LinearMap.toMatrix (EuclideanSpace.basisFun (Fin m) ℝ).toBasis
          (EuclideanSpace.basisFun (Fin p) ℝ).toBasis (Dg : (ℝ^m) →ₗ[ℝ] ℝ^p) *
        LinearMap.toMatrix (EuclideanSpace.basisFun (Fin n) ℝ).toBasis
          (EuclideanSpace.basisFun (Fin m) ℝ).toBasis (Df : (ℝ^n) →ₗ[ℝ] ℝ^m) :=
  LinearMap.toMatrix_comp _ _ _ (Dg : (ℝ^m) →ₗ[ℝ] ℝ^p) (Df : (ℝ^n) →ₗ[ℝ] ℝ^m)

/-- `thm-chain-rule`, the case of one variable: `(g ∘ f)'(a) = g'(f(a)) f'(a)`. -/
theorem chain_rule_one_variable {f g : ℝ → ℝ} {a f' g' : ℝ}
    (hf : HasDerivAt f f' a) (hg : HasDerivAt g g' (f a)) :
    HasDerivAt (g ∘ f) (g' * f') a :=
  hg.comp a hf

/-- `thm-chain-rule-paths` for three layers: entry `(i, j)` of a product of
three Jacobians is a sum, over the intermediate indices `k` and `l`, of the
products of the local partial derivatives along the path `j → l → k → i`. -/
theorem paths_three {n₀ n₁ n₂ n₃ : ℕ} (A : Matrix (Fin n₃) (Fin n₂) ℝ)
    (B : Matrix (Fin n₂) (Fin n₁) ℝ) (C : Matrix (Fin n₁) (Fin n₀) ℝ)
    (i : Fin n₃) (j : Fin n₀) :
    (A * B * C) i j = ∑ k, ∑ l, A i k * B k l * C l j := by
  simp only [Matrix.mul_apply, Finset.sum_mul]
  rw [Finset.sum_comm]

/-- `thm-product-rule`: for real-valued `f` and `g` differentiable at `a`,
`D(fg)(a) = f(a) Dg(a) + g(a) Df(a)`. -/
theorem product_rule {f g : (ℝ^n) → ℝ} {a : ℝ^n} {Df Dg : (ℝ^n) →L[ℝ] ℝ}
    (hf : HasFDerivAt f Df a) (hg : HasFDerivAt g Dg a) :
    HasFDerivAt (fun x => f x * g x) (f a • Dg + g a • Df) a :=
  hf.mul hg

/-- `thm-gradient-level-set`: if `f` is differentiable at `γ t₀` with gradient
`∇f`, the curve `γ` has velocity `γ'` at `t₀`, and `f ∘ γ` is constant near
`t₀`, then `∇f · γ' = 0`. -/
theorem gradient_level_set {f : (ℝ^n) → ℝ} {γ : ℝ → ℝ^n} {t₀ : ℝ}
    {gradf γ' : ℝ^n} (hf : HasGradientAt f gradf (γ t₀)) (hγ : HasDerivAt γ γ' t₀)
    (hc : ∀ᶠ t in nhds t₀, f (γ t) = f (γ t₀)) :
    ⟪gradf, γ'⟫_ℝ = 0 := by
  have h1 : HasDerivAt (f ∘ γ) (⟪gradf, γ'⟫_ℝ) t₀ := by
    have := (hasGradientAt_iff_hasFDerivAt.mp hf).comp_hasDerivAt t₀ hγ
    simpa [InnerProductSpace.toDual_apply_apply] using this
  have h2 : HasDerivAt (f ∘ γ) 0 t₀ :=
    (hasDerivAt_const t₀ (f (γ t₀))).congr_of_eventuallyEq
      (by filter_upwards [hc] with t ht; simpa using ht)
  exact h1.unique h2

/-- `thm-critical-point`: at a local minimum of a differentiable function,
the gradient is zero. -/
theorem critical_point_of_local_min {f : (ℝ^n) → ℝ} {a gradf : ℝ^n}
    (hmin : IsLocalMin f a) (hf : HasGradientAt f gradf a) : gradf = 0 := by
  have h := hmin.hasFDerivAt_eq_zero (hasGradientAt_iff_hasFDerivAt.mp hf)
  simpa using h

/-- `thm-mixed-partials`: a `C²` function has a symmetric second derivative,
so its mixed partial derivatives agree and its Hessian is symmetric. -/
theorem mixed_partials {f : (ℝ^n) → ℝ} {a : ℝ^n} (hf : ContDiffAt ℝ 2 f a) :
    IsSymmSndFDerivAt ℝ f a :=
  hf.isSymmSndFDerivAt (by simp)

/-- `thm-spectral-2x2`, real eigenvalues: the characteristic polynomial
`λ² - (a + d) λ + (a d - b²)` of the symmetric matrix `[[a, b], [b, d]]` has
discriminant `(a - d)² + 4 b² ≥ 0`. -/
theorem sym2_discriminant (a b d : ℝ) :
    discrim 1 (-(a + d)) (a * d - b ^ 2) = (a - d) ^ 2 + 4 * b ^ 2 ∧
      0 ≤ discrim 1 (-(a + d)) (a * d - b ^ 2) := by
  have h : discrim 1 (-(a + d)) (a * d - b ^ 2) = (a - d) ^ 2 + 4 * b ^ 2 := by
    unfold discrim; ring
  exact ⟨h, by rw [h]; positivity⟩

/-- `thm-spectral-2x2`, real eigenvalues: a real symmetric `2 × 2` matrix has
a real eigenvalue, a real root of its characteristic polynomial. -/
theorem sym2_real_eigenvalue (a b d : ℝ) :
    ∃ μ : ℝ, μ * μ - (a + d) * μ + (a * d - b ^ 2) = 0 := by
  obtain ⟨h, -⟩ := sym2_discriminant a b d
  have hs : discrim 1 (-(a + d)) (a * d - b ^ 2) =
      √((a - d) ^ 2 + 4 * b ^ 2) * √((a - d) ^ 2 + 4 * b ^ 2) := by
    rw [h, Real.mul_self_sqrt (by positivity)]
  obtain ⟨x, hx⟩ := exists_quadratic_eq_zero (a := (1 : ℝ)) (b := -(a + d))
    (c := a * d - b ^ 2) one_ne_zero ⟨_, hs⟩
  exact ⟨x, by linear_combination hx⟩

/-- `thm-spectral-2x2`, orthogonality (in any dimension): eigenvectors of a
symmetric matrix for different eigenvalues are perpendicular. -/
theorem eigenvectors_orthogonal {A : Matrix (Fin n) (Fin n) ℝ} (hA : Aᵀ = A)
    {u v : Fin n → ℝ} {μ₁ μ₂ : ℝ} (hu : A *ᵥ u = μ₁ • u) (hv : A *ᵥ v = μ₂ • v)
    (hne : μ₁ ≠ μ₂) : u ⬝ᵥ v = 0 := by
  have key : (A *ᵥ u) ⬝ᵥ v = u ⬝ᵥ (A *ᵥ v) := by
    rw [dotProduct_comm, Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, hA,
      dotProduct_comm]
  rw [hu, hv, smul_dotProduct, dotProduct_smul, smul_eq_mul, smul_eq_mul] at key
  have h : (μ₁ - μ₂) * (u ⬝ᵥ v) = 0 := by linarith
  exact (mul_eq_zero.mp h).resolve_left (sub_ne_zero.mpr hne)

end AutodiffBook.Ch02
