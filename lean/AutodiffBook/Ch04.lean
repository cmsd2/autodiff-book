import Mathlib

/-!
# Chapter 4: Reverse mode and backpropagation

Machine-checked versions of results the chapter proves by hand, named by
their labels in `chapters/04-reverse-mode.qmd`. A trace with `N` variables
is described by its matrix of local partial derivatives: `C i j` is
`∂φ_j/∂v_i` when variable `i` is an argument of operation `j`, and `0`
otherwise.
-/

open Matrix

namespace AutodiffBook.Ch04

variable {N : ℕ}

/-- `thm-forward-reverse`: if the tangents satisfy the forward recurrence
`t = s + Cᵀ t` (seed `s`) and the adjoints satisfy the reverse recurrence
`ā = w + C ā` (seed `w`), then `w ⬝ t = ā ⬝ s`. With `s = e_i` and `w = e_N`
this says that the adjoint of input `i` equals the tangent of the output in
direction `e_i`. -/
theorem forward_reverse (C : Matrix (Fin N) (Fin N) ℝ) (s w t abar : Fin N → ℝ)
    (ht : t = s + Cᵀ *ᵥ t) (ha : abar = w + C *ᵥ abar) :
    w ⬝ᵥ t = abar ⬝ᵥ s := by
  have hw : w = abar - C *ᵥ abar := eq_sub_of_add_eq ha.symm
  have hs : s = t - Cᵀ *ᵥ t := eq_sub_of_add_eq ht.symm
  have key : (C *ᵥ abar) ⬝ᵥ t = abar ⬝ᵥ (Cᵀ *ᵥ t) := by
    rw [dotProduct_comm, dotProduct_mulVec, mulVec_transpose, dotProduct_comm]
  rw [hw, hs, sub_dotProduct, dotProduct_sub, key]

/-- `thm-forward-reverse` with the seeds of `thm-reverse-mode`: seeding the
output with `1` (`w = e_o`) and the forward pass with `e_i`, the adjoint of
variable `i` equals the tangent of the output. -/
theorem adjoint_eq_tangent (C : Matrix (Fin N) (Fin N) ℝ) (i o : Fin N) (t abar : Fin N → ℝ)
    (ht : t = Pi.single i 1 + Cᵀ *ᵥ t) (ha : abar = Pi.single o 1 + C *ᵥ abar) :
    t o = abar i := by
  have h := forward_reverse C (Pi.single i 1) (Pi.single o 1) t abar ht ha
  simpa [single_dotProduct, dotProduct_single] using h

/-- `exm-ab-plus-a`: for `L = a b + a` the partial derivative with respect to `a`
at `a = 2`, `b = 3` is `b + 1 = 4`, the sum of the contributions `3` (through
the product) and `1` (directly). -/
theorem example_ab_plus_a :
    HasDerivAt (fun a : ℝ => a * 3 + a) 4 2 := by
  have h := ((hasDerivAt_id' (2 : ℝ)).mul_const 3).add (hasDerivAt_id' (2 : ℝ))
  convert h using 1
  norm_num

end AutodiffBook.Ch04
