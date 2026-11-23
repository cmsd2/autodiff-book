import Mathlib

/-!
# Chapter 1: Functions of several variables

Machine-checked versions of results the chapter proves by hand, named by
their labels in `chapters/01-several-variables.qmd`. Each
statement is phrased for `EuclideanSpace ℝ (Fin n)`, which is ℝⁿ with
the dot product and the Euclidean norm.
-/

open scoped InnerProductSpace

namespace AutodiffBook.Ch01

variable {n m : ℕ}

local notation "ℝ^" k => EuclideanSpace ℝ (Fin k)

/-- `thm-cauchy-schwarz`: the Cauchy–Schwarz inequality. -/
theorem cauchy_schwarz (x y : ℝ^n) : |⟪x, y⟫_ℝ| ≤ ‖x‖ * ‖y‖ :=
  abs_real_inner_le_norm x y

/-- `cor-triangle`: the triangle inequality. -/
theorem triangle (x y : ℝ^n) : ‖x + y‖ ≤ ‖x‖ + ‖y‖ :=
  norm_add_le x y

/-- `thm-steepest`, first half: no unit direction gives a directional
derivative larger than the norm of the gradient. -/
theorem directional_le_norm (g u : ℝ^n) (hu : ‖u‖ = 1) : ⟪g, u⟫_ℝ ≤ ‖g‖ := by
  simpa [hu] using real_inner_le_norm g u

/-- `thm-steepest`, second half: the unit vector along the gradient attains
the bound. -/
theorem directional_eq_norm (g : ℝ^n) (hg : g ≠ 0) :
    ⟪g, (‖g‖⁻¹ : ℝ) • g⟫_ℝ = ‖g‖ := by
  have h : ‖g‖ ≠ 0 := norm_ne_zero_iff.mpr hg
  rw [real_inner_smul_right, real_inner_self_eq_norm_sq]
  field_simp

/-- `thm-steepest`, the minimum: the opposite direction gives `-‖g‖`, the
smallest possible value. -/
theorem directional_ge_neg_norm (g u : ℝ^n) (hu : ‖u‖ = 1) : -‖g‖ ≤ ⟪g, u⟫_ℝ := by
  have := abs_real_inner_le_norm g u
  rw [hu, mul_one] at this
  exact neg_le_of_abs_le this

/-- `thm-differentiable-continuous`: a function that is differentiable at a point is
continuous there. -/
theorem continuousAt_of_hasFDerivAt {f : (ℝ^n) → ℝ^m} {L : (ℝ^n) →L[ℝ] ℝ^m}
    {a : ℝ^n} (h : HasFDerivAt f L a) : ContinuousAt f a :=
  h.continuousAt

/-- `exm-affine`: an affine map `x ↦ A x + b` has derivative `A` at every
point. -/
theorem affine_hasFDerivAt (A : (ℝ^n) →L[ℝ] ℝ^m) (b : ℝ^m) (a : ℝ^n) :
    HasFDerivAt (fun x => A x + b) A a :=
  A.hasFDerivAt.add_const b

end AutodiffBook.Ch01
