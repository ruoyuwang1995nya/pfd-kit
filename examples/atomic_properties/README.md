# Atomic Properties Examples

This directory contains examples demonstrating how to handle systems with special atomic properties and prepare appropriate input structure files in PFD-kit workflows. The following categories of systems are covered:

1. **Systems with atomic magnetic moments**: `fe2o3_magnetic.extxyz` and `fe2o3_magnetic_non_col.extxyz` demonstrate how to add colinear and non-colinear atomic magnetic moments to exploration systems.

2. **Systems with fixed atoms**: `si_move_mask.extxyz` demonstrates how to add constrained atoms by setting the `move_mask` property, which can be an array of either the `(natom)` or `(natom, 3)` shapes. If the move mask value can be either 0 and 1, which represent fixed atom and free atoms, respectively. Certain atoms can be held fixed during MD simulations or structural optimizations.  

