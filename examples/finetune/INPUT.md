INPUT_PARAMETERS
suffix	ABACUS
calculation	md
symmetry	0
md_type		npt
md_nstep	10
md_tfirst	900
md_dt	2
md_pfirst 0.001

kspacing  0.15
ecutwfc  100
scf_thr  1e-07
scf_nmax  100
basis_type  lcao

smearing_method  gaussian
smearing_sigma  0.01
mixing_beta  0.2
mixing_ndim  20

cal_force  1
cal_stress  1
dft_functional  PBEsol

