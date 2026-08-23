	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 6
	.section	.text._ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii,"axG",@progbits,_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii,comdat
	.protected	_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii ; -- Begin function _ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii
	.globl	_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii
	.p2align	8
	.type	_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii,@function
_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii: ; @_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii
; %bb.0:
	s_load_b128 s[16:19], s[0:1], 0x18
	s_abs_i32 s6, s2
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s3, s16, 31
	s_ashr_i32 s8, s17, 31
	s_lshr_b32 s3, s3, 24
	s_lshr_b32 s9, s8, 25
	s_add_i32 s3, s16, s3
	s_add_i32 s9, s17, s9
	s_ashr_i32 s3, s3, 8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_abs_i32 s5, s3
	s_cvt_f32_u32 s4, s5
	s_sub_i32 s7, 0, s5
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s4
	v_readfirstlane_b32 s4, v1
	s_mul_f32 s4, s4, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s4, s4
	s_mul_i32 s7, s7, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s7, s4, s7
	s_add_i32 s7, s4, s7
	s_ashr_i32 s4, s9, 7
	s_mul_hi_u32 s7, s6, s7
	s_xor_b32 s9, s2, s3
	s_mul_i32 s10, s7, s5
	s_ashr_i32 s9, s9, 31
	s_sub_i32 s6, s6, s10
	s_add_i32 s10, s7, 1
	s_sub_i32 s11, s6, s5
	s_cmp_ge_u32 s6, s5
	s_cselect_b32 s7, s10, s7
	s_cselect_b32 s6, s11, s6
	s_add_i32 s10, s7, 1
	s_cmp_ge_u32 s6, s5
	s_cselect_b32 s5, s10, s7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s6, s5, s9
	s_mov_b32 s5, 0
	s_sub_i32 s7, s6, s9
	s_lshr_b32 s6, s8, 21
	s_ashr_i32 s8, s7, 31
	s_add_i32 s9, s17, s6
	s_lshr_b32 s6, s8, 28
	s_mul_i32 s8, s7, s3
	s_add_i32 s10, s7, s6
	s_sub_i32 s6, s2, s8
	s_and_b32 s8, s10, -16
	s_ashr_i32 s2, s10, 4
	s_ashr_i32 s9, s9, 11
	s_sub_i32 s7, s7, s8
	s_cmp_ge_i32 s2, s9
	s_mul_i32 s7, s7, s3
	s_cbranch_scc0 .LBB0_2
; %bb.1:
	s_lshr_b32 s8, s4, 28
	s_add_i32 s12, s7, s6
	s_add_i32 s8, s4, s8
	s_abs_i32 s13, s12
	s_and_b32 s8, s8, -16
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_i32 s9, s4, s8
	s_abs_i32 s8, s9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_f32_u32 s10, s8
	s_sub_i32 s11, 0, s8
	v_rcp_iflag_f32_e32 v1, s10
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_3)
	v_readfirstlane_b32 s10, v1
	s_mul_f32 s10, s10, 0x4f7ffffe
	s_cvt_u32_f32 s10, s10
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s11, s11, s10
	s_mul_hi_u32 s11, s10, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_i32 s10, s10, s11
	s_xor_b32 s11, s12, s9
	s_mul_hi_u32 s10, s13, s10
	s_ashr_i32 s11, s11, 31
	s_mul_i32 s14, s10, s8
	s_sub_i32 s13, s13, s14
	s_add_i32 s14, s10, 1
	s_sub_i32 s15, s13, s8
	s_cmp_ge_u32 s13, s8
	s_cselect_b32 s10, s14, s10
	s_cselect_b32 s13, s15, s13
	s_add_i32 s14, s10, 1
	s_cmp_ge_u32 s13, s8
	s_cselect_b32 s8, s14, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s8, s8, s11
	s_sub_i32 s8, s8, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s9, s8, s9
	s_sub_i32 s9, s12, s9
	s_and_not1_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccz .LBB0_3
	s_branch .LBB0_4
.LBB0_2:
                                        ; implicit-def: $sgpr8
                                        ; implicit-def: $sgpr9
.LBB0_3:
	s_add_i32 s5, s7, s6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s6, s5, 31
	s_lshr_b32 s6, s6, 28
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_i32 s6, s5, s6
	s_and_b32 s7, s6, -16
	s_ashr_i32 s8, s6, 4
	s_sub_i32 s5, s5, s7
	s_and_b32 s6, s8, 15
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s9, s6, s5
.LBB0_4:
	s_not_b32 s5, s8
	s_lshl4_add_u32 s6, s2, s9
	s_and_b32 s2, s2, 1
	s_add_i32 s5, s3, s5
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, s8, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s2, s3
	s_cselect_b32 s5, -1, 0
	s_cmp_lt_i32 s6, s4
	s_cselect_b32 s7, -1, 0
	s_and_b32 s5, s5, s7
	s_mov_b32 s7, 0
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccnz .LBB0_6
; %bb.5:
	s_lshl_b32 s3, s3, 8
	s_lshl_b32 s4, s4, 7
	s_add_i32 s5, s3, 0xffffff00
	s_addk_i32 s4, 0xff80
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_cbranch_vccz .LBB0_7
	s_branch .LBB0_8
.LBB0_6:
                                        ; implicit-def: $sgpr4
                                        ; implicit-def: $sgpr5
.LBB0_7:
	s_lshl_b32 s5, s2, 8
	s_lshl_b32 s4, s6, 7
.LBB0_8:                                ; %_ZNK14rocm_wmma_gemm11tile_mapperILi256ELi128ELNS_8m_layoutE1ELS1_0ELi16EE8map_tileEiiiPiS3_.exit
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x10
	s_ashr_i32 s0, s5, 31
	s_ashr_i32 s6, s18, 31
	s_ashr_i32 s1, s4, 31
	s_lshr_b32 s0, s0, 24
	s_lshr_b32 s6, s6, 28
	s_lshr_b32 s1, s1, 25
	s_add_i32 s0, s5, s0
	s_add_i32 s6, s18, s6
	s_add_i32 s1, s4, s1
	s_ashr_i32 s0, s0, 8
	s_ashr_i32 s6, s6, 4
	s_ashr_i32 s7, s1, 7
	s_mul_hi_i32 s1, s6, s0
	s_mul_i32 s0, s6, s0
	v_lshlrev_b32_e32 v68, 5, v0
	s_lshl_b64 s[0:1], s[0:1], 13
	s_mul_hi_i32 s9, s6, s7
	s_mul_i32 s8, s6, s7
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s14, s0
	s_addc_u32 s1, s15, s1
	s_lshl_b64 s[8:9], s[8:9], 12
	v_lshlrev_b32_e32 v73, 4, v0
	s_add_u32 s2, s2, s8
	s_addc_u32 s3, s3, s9
	s_clause 0x1
	global_load_b128 v[3:6], v68, s[0:1]
	global_load_b128 v[7:10], v68, s[0:1] offset:16
	global_load_b128 v[11:14], v73, s[2:3]
	v_lshlrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v1, 15, v0
	s_cmp_gt_i32 s18, 31
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_lshlrev_b32_e32 v118, 2, v0
	v_and_b32_e32 v118, 16, v118
	v_xor_b32_e32 v118, v68, v118
	v_xor_b32_e32 v119, 16, v118
	v_lshlrev_b32_e32 v120, 1, v0
	v_and_b32_e32 v120, 16, v120
	v_xor_b32_e32 v120, v73, v120
	ds_store_b128 v118, v[3:6]
	ds_store_b128 v119, v[7:10]
	ds_store_b128 v120, v[11:14] offset:8192
	v_and_b32_e32 v2, 64, v2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v67, v2, v1
	s_cbranch_scc1 .LBB0_10
; %bb.9:                                ; %_ZNK14rocm_wmma_gemm11tile_mapperILi256ELi128ELNS_8m_layoutE1ELS1_0ELi16EE8map_tileEiiiPiS3_.exit.._crit_edge_crit_edge
	v_or_b32_e32 v69, v2, v1
	s_mov_b32 s7, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v65, 5, v69
	s_branch .LBB0_11
.LBB0_10:
	s_mov_b32 s7, -1
                                        ; implicit-def: $vgpr69
                                        ; implicit-def: $vgpr65
.LBB0_11:                               ; %Flow1115
	v_mov_b32_e32 v8, 0
	v_and_b32_e32 v66, 0x19e0, v68
	s_and_not1_b32 vcc_lo, exec_lo, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v7, v8
	v_mov_b32_e32 v6, v8
	v_mov_b32_e32 v5, v8
	v_mov_b32_e32 v4, v8
	v_mov_b32_e32 v3, v8
	v_mov_b32_e32 v2, v8
	v_mov_b32_e32 v1, v8
	v_mov_b32_e32 v16, v8
	v_mov_b32_e32 v15, v8
	v_mov_b32_e32 v14, v8
	v_mov_b32_e32 v13, v8
	v_mov_b32_e32 v12, v8
	v_mov_b32_e32 v11, v8
	v_mov_b32_e32 v10, v8
	v_mov_b32_e32 v9, v8
	v_mov_b32_e32 v24, v8
	v_mov_b32_e32 v23, v8
	v_mov_b32_e32 v22, v8
	v_mov_b32_e32 v21, v8
	v_mov_b32_e32 v20, v8
	v_mov_b32_e32 v19, v8
	v_mov_b32_e32 v18, v8
	v_mov_b32_e32 v17, v8
	v_mov_b32_e32 v32, v8
	v_mov_b32_e32 v31, v8
	v_mov_b32_e32 v30, v8
	v_mov_b32_e32 v29, v8
	v_mov_b32_e32 v28, v8
	v_mov_b32_e32 v27, v8
	v_mov_b32_e32 v26, v8
	v_mov_b32_e32 v25, v8
	v_mov_b32_e32 v40, v8
	v_mov_b32_e32 v39, v8
	v_mov_b32_e32 v38, v8
	v_mov_b32_e32 v37, v8
	v_mov_b32_e32 v36, v8
	v_mov_b32_e32 v35, v8
	v_mov_b32_e32 v34, v8
	v_mov_b32_e32 v33, v8
	v_mov_b32_e32 v48, v8
	v_mov_b32_e32 v47, v8
	v_mov_b32_e32 v46, v8
	v_mov_b32_e32 v45, v8
	v_mov_b32_e32 v44, v8
	v_mov_b32_e32 v43, v8
	v_mov_b32_e32 v42, v8
	v_mov_b32_e32 v41, v8
	v_mov_b32_e32 v56, v8
	v_mov_b32_e32 v55, v8
	v_mov_b32_e32 v54, v8
	v_mov_b32_e32 v53, v8
	v_mov_b32_e32 v52, v8
	v_mov_b32_e32 v51, v8
	v_mov_b32_e32 v50, v8
	v_mov_b32_e32 v49, v8
	v_mov_b32_e32 v64, v8
	v_mov_b32_e32 v63, v8
	v_mov_b32_e32 v62, v8
	v_mov_b32_e32 v61, v8
	v_mov_b32_e32 v60, v8
	v_mov_b32_e32 v59, v8
	v_mov_b32_e32 v58, v8
	v_mov_b32_e32 v57, v8
	s_cbranch_vccnz .LBB0_15
; %bb.12:                               ; %.lr.ph
	v_mov_b32_e32 v57, 0
	v_add_co_u32 v69, s0, s0, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v70, null, s1, 0, s0
	v_add_co_u32 v71, s0, s2, v73
	v_add_co_ci_u32_e64 v72, null, s3, 0, s0
	v_or_b32_e32 v73, 0x2000, v73
	v_or_b32_e32 v120, 0x2000, v120
	v_dual_mov_b32 v58, v57 :: v_dual_lshlrev_b32 v65, 5, v67
	v_lshlrev_b32_e32 v121, 2, v67
	v_and_b32_e32 v121, 16, v121
	v_xor_b32_e32 v66, v66, v121
	v_xor_b32_e32 v122, 16, v66
	v_xor_b32_e32 v65, v65, v121
	v_xor_b32_e32 v121, 16, v65
	v_mov_b32_e32 v59, v57
	v_mov_b32_e32 v60, v57
	v_mov_b32_e32 v61, v57
	v_mov_b32_e32 v62, v57
	v_mov_b32_e32 v63, v57
	v_mov_b32_e32 v64, v57
	v_mov_b32_e32 v49, v57
	v_mov_b32_e32 v50, v57
	v_mov_b32_e32 v51, v57
	v_mov_b32_e32 v52, v57
	v_mov_b32_e32 v53, v57
	v_mov_b32_e32 v54, v57
	v_mov_b32_e32 v55, v57
	v_mov_b32_e32 v56, v57
	v_mov_b32_e32 v41, v57
	v_mov_b32_e32 v42, v57
	v_mov_b32_e32 v43, v57
	v_mov_b32_e32 v44, v57
	v_mov_b32_e32 v45, v57
	v_mov_b32_e32 v46, v57
	v_mov_b32_e32 v47, v57
	v_mov_b32_e32 v48, v57
	v_mov_b32_e32 v33, v57
	v_mov_b32_e32 v34, v57
	v_mov_b32_e32 v35, v57
	v_mov_b32_e32 v36, v57
	v_mov_b32_e32 v37, v57
	v_mov_b32_e32 v38, v57
	v_mov_b32_e32 v39, v57
	v_mov_b32_e32 v40, v57
	v_mov_b32_e32 v25, v57
	v_mov_b32_e32 v26, v57
	v_mov_b32_e32 v27, v57
	v_mov_b32_e32 v28, v57
	v_mov_b32_e32 v29, v57
	v_mov_b32_e32 v30, v57
	v_mov_b32_e32 v31, v57
	v_mov_b32_e32 v32, v57
	v_mov_b32_e32 v17, v57
	v_mov_b32_e32 v18, v57
	v_mov_b32_e32 v19, v57
	v_mov_b32_e32 v20, v57
	v_mov_b32_e32 v21, v57
	v_mov_b32_e32 v22, v57
	v_mov_b32_e32 v23, v57
	v_mov_b32_e32 v24, v57
	v_mov_b32_e32 v9, v57
	v_mov_b32_e32 v10, v57
	v_mov_b32_e32 v11, v57
	v_mov_b32_e32 v12, v57
	v_mov_b32_e32 v13, v57
	v_mov_b32_e32 v14, v57
	v_mov_b32_e32 v15, v57
	v_mov_b32_e32 v16, v57
	v_mov_b32_e32 v1, v57
	v_mov_b32_e32 v2, v57
	v_mov_b32_e32 v3, v57
	v_mov_b32_e32 v4, v57
	v_mov_b32_e32 v5, v57
	v_mov_b32_e32 v6, v57
	v_mov_b32_e32 v7, v57
	v_mov_b32_e32 v8, v57
	s_max_i32 s0, s6, 2
	s_movk_i32 s2, 0x1000
	s_add_i32 s6, s0, -1
	s_movk_i32 s0, 0x800
	s_mov_b32 s3, 0
.LBB0_13:                               ; =>This Inner Loop Header: Depth=1
	ds_load_b128 v[74:77], v66
	ds_load_b128 v[78:81], v122
	ds_load_b128 v[86:89], v122 offset:512
	ds_load_b128 v[82:85], v66 offset:512
	ds_load_b128 v[90:93], v65 offset:8192
	ds_load_b128 v[94:97], v121 offset:8192
	ds_load_b128 v[102:105], v122 offset:1024
	ds_load_b128 v[98:101], v66 offset:1024
	ds_load_b128 v[110:113], v122 offset:1536
	ds_load_b128 v[106:109], v66 offset:1536
	s_lshl_b64 s[8:9], s[2:3], 1
	s_mov_b32 s1, s3
	v_add_co_u32 v114, vcc_lo, v69, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v115, null, s9, v70, vcc_lo
	s_lshl_b64 s[8:9], s[0:1], 1
	s_add_i32 s6, s6, -1
	v_add_co_u32 v116, vcc_lo, v71, s8
	v_add_co_ci_u32_e64 v117, null, s9, v72, vcc_lo
	s_addk_i32 s2, 0x1000
	s_addk_i32 s0, 0x800
	s_cmp_eq_u32 s6, 0
	s_waitcnt lgkmcnt(4)
	v_wmma_f16_16x16x16_f16 v[57:64], v[74:81], v[90:97], v[57:64]
	v_wmma_f16_16x16x16_f16 v[41:48], v[82:89], v[90:97], v[41:48]
	s_waitcnt lgkmcnt(2)
	v_wmma_f16_16x16x16_f16 v[25:32], v[98:105], v[90:97], v[25:32]
	s_waitcnt lgkmcnt(0)
	v_wmma_f16_16x16x16_f16 v[9:16], v[106:113], v[90:97], v[9:16]
	ds_load_b128 v[94:97], v121 offset:8704
	ds_load_b128 v[90:93], v65 offset:8704
	s_waitcnt lgkmcnt(0)
	v_wmma_f16_16x16x16_f16 v[49:56], v[74:81], v[90:97], v[49:56]
	v_wmma_f16_16x16x16_f16 v[33:40], v[82:89], v[90:97], v[33:40]
	v_wmma_f16_16x16x16_f16 v[17:24], v[98:105], v[90:97], v[17:24]
	v_wmma_f16_16x16x16_f16 v[1:8], v[106:113], v[90:97], v[1:8]
	ds_load_b128 v[94:97], v121 offset:9216
	ds_load_b128 v[90:93], v65 offset:9216
	s_waitcnt lgkmcnt(0)
	v_wmma_f16_16x16x16_f16 v[57:64], v[74:81], v[90:97], v[57:64] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[41:48], v[82:89], v[90:97], v[41:48] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[25:32], v[98:105], v[90:97], v[25:32] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[9:16], v[106:113], v[90:97], v[9:16] op_sel:[0,0,1]
	ds_load_b128 v[94:97], v121 offset:9728
	ds_load_b128 v[90:93], v65 offset:9728
	s_waitcnt lgkmcnt(0)
	v_wmma_f16_16x16x16_f16 v[49:56], v[74:81], v[90:97], v[49:56] op_sel:[0,0,1]
	s_clause 0x1
	global_load_b128 v[74:77], v[114:115], off
	global_load_b128 v[78:81], v[114:115], off offset:16
	global_load_b128 v[114:117], v[116:117], off
	v_wmma_f16_16x16x16_f16 v[33:40], v[82:89], v[90:97], v[33:40] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[17:24], v[98:105], v[90:97], v[17:24] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[1:8], v[106:113], v[90:97], v[1:8] op_sel:[0,0,1]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(0)
	s_barrier
	ds_store_b128 v118, v[74:77]
	ds_store_b128 v119, v[78:81]
	ds_store_b128 v120, v[114:117]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_cbranch_scc0 .LBB0_13
; %bb.14:                               ; %Flow
	v_mov_b32_e32 v69, v67
.LBB0_15:                               ; %Flow1116
	ds_load_b128 v[70:73], v66
	ds_load_b128 v[74:77], v122
	ds_load_b128 v[82:85], v122 offset:512
	ds_load_b128 v[78:81], v66 offset:512
	ds_load_b128 v[86:89], v65 offset:8192
	ds_load_b128 v[90:93], v121 offset:8192
	ds_load_b128 v[98:101], v122 offset:1024
	ds_load_b128 v[94:97], v66 offset:1024
	ds_load_b128 v[106:109], v122 offset:1536
	ds_load_b128 v[102:105], v66 offset:1536
	ds_load_b128 v[114:117], v121 offset:8704
	ds_load_b128 v[110:113], v65 offset:8704
	s_waitcnt lgkmcnt(6)
	v_wmma_f16_16x16x16_f16 v[57:64], v[70:77], v[86:93], v[57:64]
	v_wmma_f16_16x16x16_f16 v[41:48], v[78:85], v[86:93], v[41:48]
	s_waitcnt lgkmcnt(4)
	v_wmma_f16_16x16x16_f16 v[25:32], v[94:101], v[86:93], v[25:32]
	s_waitcnt lgkmcnt(2)
	v_wmma_f16_16x16x16_f16 v[9:16], v[102:109], v[86:93], v[9:16]
	ds_load_b128 v[90:93], v121 offset:9216
	ds_load_b128 v[86:89], v65 offset:9216
	s_waitcnt lgkmcnt(2)
	v_wmma_f16_16x16x16_f16 v[49:56], v[70:77], v[110:117], v[49:56]
	v_wmma_f16_16x16x16_f16 v[33:40], v[78:85], v[110:117], v[33:40]
	v_wmma_f16_16x16x16_f16 v[17:24], v[94:101], v[110:117], v[17:24]
	v_wmma_f16_16x16x16_f16 v[1:8], v[102:109], v[110:117], v[1:8]
	ds_load_b128 v[114:117], v121 offset:9728
	ds_load_b128 v[110:113], v65 offset:9728
	v_and_b32_e32 v65, 0xc0, v0
	v_lshrrev_b32_e32 v0, 4, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v0, v0, 1, v65
	v_add_nc_u32_e32 v65, s5, v0
	v_add_nc_u32_e32 v0, s4, v69
	s_waitcnt lgkmcnt(2)
	v_wmma_f16_16x16x16_f16 v[57:64], v[70:77], v[86:93], v[57:64] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[41:48], v[78:85], v[86:93], v[41:48] op_sel:[0,0,1]
	v_mul_lo_u32 v68, v65, s17
	v_cmp_gt_i32_e64 s3, s16, v65
	v_cmp_gt_i32_e32 vcc_lo, s17, v0
	v_wmma_f16_16x16x16_f16 v[25:32], v[94:101], v[86:93], v[25:32] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[9:16], v[102:109], v[86:93], v[9:16] op_sel:[0,0,1]
	s_waitcnt lgkmcnt(0)
	v_wmma_f16_16x16x16_f16 v[49:56], v[70:77], v[110:117], v[49:56] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[33:40], v[78:85], v[110:117], v[33:40] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[17:24], v[94:101], v[110:117], v[17:24] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[1:8], v[102:109], v[110:117], v[1:8] op_sel:[0,0,1]
	s_and_b32 s0, s3, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_17
; %bb.16:
	v_add_nc_u32_e32 v66, v0, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s12, v66
	v_add_co_ci_u32_e64 v67, null, s13, v67, s0
	global_store_b16 v[66:67], v57, off
.LBB0_17:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 2, v65
	s_lshl_b32 s14, s17, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v69, s14, v68
	v_cmp_gt_i32_e64 s4, s16, v66
	s_and_b32 s0, s4, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_19
; %bb.18:
	v_add_nc_u32_e32 v66, v0, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s12, v66
	v_add_co_ci_u32_e64 v67, null, s13, v67, s0
	global_store_b16 v[66:67], v58, off
.LBB0_19:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 4, v65
	v_add_nc_u32_e32 v70, s14, v69
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s5, s16, v66
	s_and_b32 s0, s5, vcc_lo
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_21
; %bb.20:
	v_add_nc_u32_e32 v66, v0, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s12, v66
	v_add_co_ci_u32_e64 v67, null, s13, v67, s0
	global_store_b16 v[66:67], v59, off
.LBB0_21:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 6, v65
	v_add_nc_u32_e32 v72, s14, v70
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s6, s16, v66
	s_and_b32 s0, s6, vcc_lo
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_23
; %bb.22:
	v_add_nc_u32_e32 v66, v0, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s12, v66
	v_add_co_ci_u32_e64 v67, null, s13, v67, s0
	global_store_b16 v[66:67], v60, off
.LBB0_23:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 8, v65
	v_add_nc_u32_e32 v73, s14, v72
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s7, s16, v66
	s_and_b32 s0, s7, vcc_lo
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_25
; %bb.24:
	v_add_nc_u32_e32 v66, v0, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s12, v66
	v_add_co_ci_u32_e64 v67, null, s13, v67, s0
	global_store_b16 v[66:67], v61, off
.LBB0_25:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 10, v65
	v_add_nc_u32_e32 v74, s14, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s8, s16, v66
	s_and_b32 s0, s8, vcc_lo
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_27
; %bb.26:
	v_add_nc_u32_e32 v66, v0, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s12, v66
	v_add_co_ci_u32_e64 v67, null, s13, v67, s0
	global_store_b16 v[66:67], v62, off
.LBB0_27:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 12, v65
	v_add_nc_u32_e32 v75, s14, v74
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s16, v66
	s_and_b32 s0, s9, vcc_lo
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_29
; %bb.28:
	v_add_nc_u32_e32 v66, v0, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s12, v66
	v_add_co_ci_u32_e64 v67, null, s13, v67, s0
	global_store_b16 v[66:67], v63, off
.LBB0_29:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 14, v65
	v_add_nc_u32_e32 v71, s14, v75
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s10, s16, v66
	s_and_b32 s0, s10, vcc_lo
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_31
; %bb.30:
	v_add_nc_u32_e32 v66, v0, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s12, v66
	v_add_co_ci_u32_e64 v67, null, s13, v67, s0
	global_store_b16 v[66:67], v64, off
.LBB0_31:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 16, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s0, s17, v66
	s_and_b32 s1, s3, s0
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_178
; %bb.32:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_179
.LBB0_33:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_180
.LBB0_34:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_181
.LBB0_35:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_182
.LBB0_36:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_183
.LBB0_37:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_184
.LBB0_38:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_40
.LBB0_39:
	v_add_nc_u32_e32 v76, v66, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s12, v76
	v_add_co_ci_u32_e64 v77, null, s13, v77, s1
	global_store_b16 v[76:77], v56, off
.LBB0_40:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.1
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v67, 32, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s1, s17, v67
	s_and_b32 s2, s3, s1
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_185
; %bb.41:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_186
.LBB0_42:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_187
.LBB0_43:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_188
.LBB0_44:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_189
.LBB0_45:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_190
.LBB0_46:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_191
.LBB0_47:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_49
.LBB0_48:
	v_add_nc_u32_e32 v57, v67, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64 v[57:58], 1, v[57:58]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v57, s2, s12, v57
	v_add_co_ci_u32_e64 v58, null, s13, v58, s2
	global_store_d16_hi_b16 v[57:58], v64, off
.LBB0_49:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.2
	s_or_b32 exec_lo, exec_lo, s11
	v_add_nc_u32_e32 v57, 48, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s2, s17, v57
	s_and_b32 s3, s3, s2
	s_and_saveexec_b32 s11, s3
	s_cbranch_execnz .LBB0_192
; %bb.50:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s3, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_193
.LBB0_51:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_194
.LBB0_52:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_195
.LBB0_53:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_196
.LBB0_54:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_197
.LBB0_55:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_198
.LBB0_56:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_58
.LBB0_57:
	v_add_nc_u32_e32 v49, v57, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v50, 31, v49
	v_lshlrev_b64 v[49:50], 1, v[49:50]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v49, s3, s12, v49
	v_add_co_ci_u32_e64 v50, null, s13, v50, s3
	global_store_d16_hi_b16 v[49:50], v56, off
.LBB0_58:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.3
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v49, 16, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s3, s16, v49
	v_add_nc_u32_e32 v49, s14, v71
	s_and_b32 s4, s3, vcc_lo
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB0_60
; %bb.59:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v50, v0, v49
	v_ashrrev_i32_e32 v51, 31, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[50:51], 1, v[50:51]
	v_add_co_u32 v50, s4, s12, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v51, null, s13, v51, s4
	global_store_b16 v[50:51], v41, off
.LBB0_60:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.1210
	s_or_b32 exec_lo, exec_lo, s5
	v_add_nc_u32_e32 v50, 18, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s4, s16, v50
	v_add_nc_u32_e32 v50, s14, v49
	s_and_b32 s5, s4, vcc_lo
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB0_62
; %bb.61:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v51, v0, v50
	v_ashrrev_i32_e32 v52, 31, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[51:52], 1, v[51:52]
	v_add_co_u32 v51, s5, s12, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v52, null, s13, v52, s5
	global_store_b16 v[51:52], v42, off
.LBB0_62:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.1212
	s_or_b32 exec_lo, exec_lo, s6
	v_add_nc_u32_e32 v51, 20, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s5, s16, v51
	v_add_nc_u32_e32 v51, s14, v50
	s_and_b32 s6, s5, vcc_lo
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_64
; %bb.63:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v52, v0, v51
	v_ashrrev_i32_e32 v53, 31, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[52:53], 1, v[52:53]
	v_add_co_u32 v52, s6, s12, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v53, null, s13, v53, s6
	global_store_b16 v[52:53], v43, off
.LBB0_64:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.1214
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v52, 22, v65
	v_add_nc_u32_e32 v53, s14, v51
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s6, s16, v52
	s_and_b32 s7, s6, vcc_lo
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_66
; %bb.65:
	v_add_nc_u32_e32 v54, v0, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v55, 31, v54
	v_lshlrev_b64 v[54:55], 1, v[54:55]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v54, s7, s12, v54
	v_add_co_ci_u32_e64 v55, null, s13, v55, s7
	global_store_b16 v[54:55], v44, off
.LBB0_66:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.1216
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v52, 24, v65
	v_add_nc_u32_e32 v54, s14, v53
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s7, s16, v52
	s_and_b32 s8, s7, vcc_lo
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB0_68
; %bb.67:
	v_add_nc_u32_e32 v55, v0, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v56, 31, v55
	v_lshlrev_b64 v[55:56], 1, v[55:56]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v55, s8, s12, v55
	v_add_co_ci_u32_e64 v56, null, s13, v56, s8
	global_store_b16 v[55:56], v45, off
.LBB0_68:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.1218
	s_or_b32 exec_lo, exec_lo, s9
	v_add_nc_u32_e32 v52, 26, v65
	v_add_nc_u32_e32 v55, s14, v54
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s8, s16, v52
	s_and_b32 s9, s8, vcc_lo
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB0_70
; %bb.69:
	v_add_nc_u32_e32 v58, v0, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s9, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s9
	global_store_b16 v[58:59], v46, off
.LBB0_70:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.1220
	s_or_b32 exec_lo, exec_lo, s10
	v_add_nc_u32_e32 v52, 28, v65
	v_add_nc_u32_e32 v56, s14, v55
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s16, v52
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB0_72
; %bb.71:
	v_add_nc_u32_e32 v58, v0, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s10, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s10
	global_store_b16 v[58:59], v47, off
.LBB0_72:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.1222
	s_or_b32 exec_lo, exec_lo, s11
	v_add_nc_u32_e32 v52, 30, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s10, s16, v52
	v_add_nc_u32_e32 v52, s14, v56
	s_and_b32 s11, s10, vcc_lo
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_199
; %bb.73:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.1223
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_200
.LBB0_74:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_201
.LBB0_75:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_202
.LBB0_76:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_203
.LBB0_77:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_204
.LBB0_78:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_205
.LBB0_79:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_206
.LBB0_80:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_207
.LBB0_81:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_208
.LBB0_82:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_209
.LBB0_83:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_210
.LBB0_84:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_211
.LBB0_85:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_212
.LBB0_86:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_213
.LBB0_87:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_214
.LBB0_88:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_215
.LBB0_89:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s3, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s3
	s_cbranch_execnz .LBB0_216
.LBB0_90:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s3, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_217
.LBB0_91:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_218
.LBB0_92:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_219
.LBB0_93:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_220
.LBB0_94:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_221
.LBB0_95:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_222
.LBB0_96:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_98
.LBB0_97:
	v_add_nc_u32_e32 v33, v57, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v34, 31, v33
	v_lshlrev_b64 v[33:34], 1, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s3, s12, v33
	v_add_co_ci_u32_e64 v34, null, s13, v34, s3
	global_store_d16_hi_b16 v[33:34], v40, off
.LBB0_98:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.3.1
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v33, 32, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s3, s16, v33
	v_add_nc_u32_e32 v33, s14, v52
	s_and_b32 s4, s3, vcc_lo
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB0_100
; %bb.99:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v34, v0, v33
	v_ashrrev_i32_e32 v35, 31, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[34:35], 1, v[34:35]
	v_add_co_u32 v34, s4, s12, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v35, null, s13, v35, s4
	global_store_b16 v[34:35], v25, off
.LBB0_100:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.2242
	s_or_b32 exec_lo, exec_lo, s5
	v_add_nc_u32_e32 v34, 34, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s4, s16, v34
	v_add_nc_u32_e32 v34, s14, v33
	s_and_b32 s5, s4, vcc_lo
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB0_102
; %bb.101:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v35, v0, v34
	v_ashrrev_i32_e32 v36, 31, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[35:36], 1, v[35:36]
	v_add_co_u32 v35, s5, s12, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v36, null, s13, v36, s5
	global_store_b16 v[35:36], v26, off
.LBB0_102:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.2244
	s_or_b32 exec_lo, exec_lo, s6
	v_add_nc_u32_e32 v35, 36, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s5, s16, v35
	v_add_nc_u32_e32 v35, s14, v34
	s_and_b32 s6, s5, vcc_lo
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_104
; %bb.103:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v36, v0, v35
	v_ashrrev_i32_e32 v37, 31, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[36:37], 1, v[36:37]
	v_add_co_u32 v36, s6, s12, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v37, null, s13, v37, s6
	global_store_b16 v[36:37], v27, off
.LBB0_104:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.2246
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v36, 38, v65
	v_add_nc_u32_e32 v37, s14, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s6, s16, v36
	s_and_b32 s7, s6, vcc_lo
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_106
; %bb.105:
	v_add_nc_u32_e32 v38, v0, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v39, 31, v38
	v_lshlrev_b64 v[38:39], 1, v[38:39]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v38, s7, s12, v38
	v_add_co_ci_u32_e64 v39, null, s13, v39, s7
	global_store_b16 v[38:39], v28, off
.LBB0_106:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.2248
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v36, 40, v65
	v_add_nc_u32_e32 v38, s14, v37
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s7, s16, v36
	s_and_b32 s8, s7, vcc_lo
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB0_108
; %bb.107:
	v_add_nc_u32_e32 v39, v0, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v40, 31, v39
	v_lshlrev_b64 v[39:40], 1, v[39:40]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v39, s8, s12, v39
	v_add_co_ci_u32_e64 v40, null, s13, v40, s8
	global_store_b16 v[39:40], v29, off
.LBB0_108:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.2250
	s_or_b32 exec_lo, exec_lo, s9
	v_add_nc_u32_e32 v36, 42, v65
	v_add_nc_u32_e32 v39, s14, v38
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s8, s16, v36
	s_and_b32 s9, s8, vcc_lo
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB0_110
; %bb.109:
	v_add_nc_u32_e32 v40, v0, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v41, 31, v40
	v_lshlrev_b64 v[40:41], 1, v[40:41]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v40, s9, s12, v40
	v_add_co_ci_u32_e64 v41, null, s13, v41, s9
	global_store_b16 v[40:41], v30, off
.LBB0_110:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.2252
	s_or_b32 exec_lo, exec_lo, s10
	v_add_nc_u32_e32 v36, 44, v65
	v_add_nc_u32_e32 v40, s14, v39
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s16, v36
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB0_112
; %bb.111:
	v_add_nc_u32_e32 v41, v0, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s10, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s10
	global_store_b16 v[41:42], v31, off
.LBB0_112:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.2254
	s_or_b32 exec_lo, exec_lo, s11
	v_add_nc_u32_e32 v36, 46, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s10, s16, v36
	v_add_nc_u32_e32 v36, s14, v40
	s_and_b32 s11, s10, vcc_lo
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_223
; %bb.113:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.2255
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_224
.LBB0_114:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_225
.LBB0_115:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_226
.LBB0_116:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_227
.LBB0_117:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_228
.LBB0_118:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_229
.LBB0_119:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_230
.LBB0_120:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_231
.LBB0_121:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_232
.LBB0_122:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_233
.LBB0_123:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_234
.LBB0_124:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_235
.LBB0_125:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_236
.LBB0_126:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_237
.LBB0_127:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_238
.LBB0_128:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_239
.LBB0_129:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s3, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s3
	s_cbranch_execnz .LBB0_240
.LBB0_130:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s3, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_241
.LBB0_131:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_242
.LBB0_132:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_243
.LBB0_133:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_244
.LBB0_134:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_245
.LBB0_135:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_246
.LBB0_136:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_138
.LBB0_137:
	v_add_nc_u32_e32 v17, v57, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v17, s3, s12, v17
	v_add_co_ci_u32_e64 v18, null, s13, v18, s3
	global_store_d16_hi_b16 v[17:18], v24, off
.LBB0_138:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.3.2
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v17, 48, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s3, s16, v17
	v_add_nc_u32_e32 v17, s14, v36
	s_and_b32 s4, s3, vcc_lo
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB0_140
; %bb.139:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v18, v0, v17
	v_ashrrev_i32_e32 v19, 31, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[18:19], 1, v[18:19]
	v_add_co_u32 v18, s4, s12, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v19, null, s13, v19, s4
	global_store_b16 v[18:19], v9, off
.LBB0_140:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.3274
	s_or_b32 exec_lo, exec_lo, s5
	v_add_nc_u32_e32 v18, 50, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s4, s16, v18
	v_add_nc_u32_e32 v18, s14, v17
	s_and_b32 s5, s4, vcc_lo
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB0_142
; %bb.141:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v19, v0, v18
	v_ashrrev_i32_e32 v20, 31, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[19:20], 1, v[19:20]
	v_add_co_u32 v19, s5, s12, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v20, null, s13, v20, s5
	global_store_b16 v[19:20], v10, off
.LBB0_142:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.3276
	s_or_b32 exec_lo, exec_lo, s6
	v_add_nc_u32_e32 v19, 52, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s5, s16, v19
	v_add_nc_u32_e32 v19, s14, v18
	s_and_b32 s6, s5, vcc_lo
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_144
; %bb.143:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v20, v0, v19
	v_ashrrev_i32_e32 v21, 31, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[20:21], 1, v[20:21]
	v_add_co_u32 v20, s6, s12, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v21, null, s13, v21, s6
	global_store_b16 v[20:21], v11, off
.LBB0_144:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.3278
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v20, 54, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s6, s16, v20
	v_add_nc_u32_e32 v20, s14, v19
	s_and_b32 s7, s6, vcc_lo
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_146
; %bb.145:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v21, v0, v20
	v_ashrrev_i32_e32 v22, 31, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[21:22], 1, v[21:22]
	v_add_co_u32 v21, s7, s12, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v22, null, s13, v22, s7
	global_store_b16 v[21:22], v12, off
.LBB0_146:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.3280
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v21, 56, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s7, s16, v21
	v_add_nc_u32_e32 v21, s14, v20
	s_and_b32 s8, s7, vcc_lo
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB0_148
; %bb.147:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v22, v0, v21
	v_ashrrev_i32_e32 v23, 31, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[22:23], 1, v[22:23]
	v_add_co_u32 v22, s8, s12, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v23, null, s13, v23, s8
	global_store_b16 v[22:23], v13, off
.LBB0_148:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.3282
	s_or_b32 exec_lo, exec_lo, s9
	v_add_nc_u32_e32 v22, 58, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s8, s16, v22
	v_add_nc_u32_e32 v22, s14, v21
	s_and_b32 s9, s8, vcc_lo
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB0_150
; %bb.149:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v23, v0, v22
	v_ashrrev_i32_e32 v24, 31, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[23:24], 1, v[23:24]
	v_add_co_u32 v23, s9, s12, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v24, null, s13, v24, s9
	global_store_b16 v[23:24], v14, off
.LBB0_150:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.3284
	s_or_b32 exec_lo, exec_lo, s10
	v_add_nc_u32_e32 v23, 60, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s16, v23
	v_add_nc_u32_e32 v23, s14, v22
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB0_152
; %bb.151:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v0, v23
	v_ashrrev_i32_e32 v25, 31, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[24:25], 1, v[24:25]
	v_add_co_u32 v24, s10, s12, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v25, null, s13, v25, s10
	global_store_b16 v[24:25], v15, off
.LBB0_152:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.3286
	s_or_b32 exec_lo, exec_lo, s11
	v_add_nc_u32_e32 v24, 62, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s10, s16, v24
	v_add_nc_u32_e32 v24, s14, v23
	s_and_b32 s14, s10, vcc_lo
	s_and_saveexec_b32 s11, s14
	s_cbranch_execnz .LBB0_247
; %bb.153:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.3287
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s3, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execnz .LBB0_248
.LBB0_154:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execnz .LBB0_249
.LBB0_155:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execnz .LBB0_250
.LBB0_156:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execnz .LBB0_251
.LBB0_157:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execnz .LBB0_252
.LBB0_158:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execnz .LBB0_253
.LBB0_159:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execnz .LBB0_254
.LBB0_160:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_255
.LBB0_161:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.1.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_256
.LBB0_162:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_257
.LBB0_163:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_258
.LBB0_164:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_259
.LBB0_165:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_260
.LBB0_166:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_261
.LBB0_167:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_262
.LBB0_168:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_263
.LBB0_169:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_264
.LBB0_170:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_265
.LBB0_171:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_266
.LBB0_172:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_267
.LBB0_173:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_268
.LBB0_174:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_269
.LBB0_175:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_270
.LBB0_176:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s0, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execnz .LBB0_271
.LBB0_177:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.3.3
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.LBB0_178:
	v_add_nc_u32_e32 v76, v66, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s12, v76
	v_add_co_ci_u32_e64 v77, null, s13, v77, s1
	global_store_b16 v[76:77], v49, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_33
.LBB0_179:
	v_add_nc_u32_e32 v76, v66, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s12, v76
	v_add_co_ci_u32_e64 v77, null, s13, v77, s1
	global_store_b16 v[76:77], v50, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_34
.LBB0_180:
	v_add_nc_u32_e32 v76, v66, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s12, v76
	v_add_co_ci_u32_e64 v77, null, s13, v77, s1
	global_store_b16 v[76:77], v51, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_35
.LBB0_181:
	v_add_nc_u32_e32 v76, v66, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s12, v76
	v_add_co_ci_u32_e64 v77, null, s13, v77, s1
	global_store_b16 v[76:77], v52, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_36
.LBB0_182:
	v_add_nc_u32_e32 v76, v66, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s12, v76
	v_add_co_ci_u32_e64 v77, null, s13, v77, s1
	global_store_b16 v[76:77], v53, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_37
.LBB0_183:
	v_add_nc_u32_e32 v76, v66, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s12, v76
	v_add_co_ci_u32_e64 v77, null, s13, v77, s1
	global_store_b16 v[76:77], v54, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_38
.LBB0_184:
	v_add_nc_u32_e32 v76, v66, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s12, v76
	v_add_co_ci_u32_e64 v77, null, s13, v77, s1
	global_store_b16 v[76:77], v55, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_39
	s_branch .LBB0_40
.LBB0_185:
	v_add_nc_u32_e32 v76, v67, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s2, s12, v76
	v_add_co_ci_u32_e64 v77, null, s13, v77, s2
	global_store_d16_hi_b16 v[76:77], v57, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_42
.LBB0_186:
	v_add_nc_u32_e32 v76, v67, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s2, s12, v76
	v_add_co_ci_u32_e64 v77, null, s13, v77, s2
	global_store_d16_hi_b16 v[76:77], v58, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_43
.LBB0_187:
	v_add_nc_u32_e32 v57, v67, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64 v[57:58], 1, v[57:58]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v57, s2, s12, v57
	v_add_co_ci_u32_e64 v58, null, s13, v58, s2
	global_store_d16_hi_b16 v[57:58], v59, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_44
.LBB0_188:
	v_add_nc_u32_e32 v57, v67, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64 v[57:58], 1, v[57:58]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v57, s2, s12, v57
	v_add_co_ci_u32_e64 v58, null, s13, v58, s2
	global_store_d16_hi_b16 v[57:58], v60, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_45
.LBB0_189:
	v_add_nc_u32_e32 v57, v67, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64 v[57:58], 1, v[57:58]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v57, s2, s12, v57
	v_add_co_ci_u32_e64 v58, null, s13, v58, s2
	global_store_d16_hi_b16 v[57:58], v61, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_46
.LBB0_190:
	v_add_nc_u32_e32 v57, v67, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64 v[57:58], 1, v[57:58]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v57, s2, s12, v57
	v_add_co_ci_u32_e64 v58, null, s13, v58, s2
	global_store_d16_hi_b16 v[57:58], v62, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_47
.LBB0_191:
	v_add_nc_u32_e32 v57, v67, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64 v[57:58], 1, v[57:58]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v57, s2, s12, v57
	v_add_co_ci_u32_e64 v58, null, s13, v58, s2
	global_store_d16_hi_b16 v[57:58], v63, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_48
	s_branch .LBB0_49
.LBB0_192:
	v_add_nc_u32_e32 v58, v57, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s3, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s3
	global_store_d16_hi_b16 v[58:59], v49, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s3, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_51
.LBB0_193:
	v_add_nc_u32_e32 v58, v57, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s3, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s3
	global_store_d16_hi_b16 v[58:59], v50, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_52
.LBB0_194:
	v_add_nc_u32_e32 v49, v57, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v50, 31, v49
	v_lshlrev_b64 v[49:50], 1, v[49:50]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v49, s3, s12, v49
	v_add_co_ci_u32_e64 v50, null, s13, v50, s3
	global_store_d16_hi_b16 v[49:50], v51, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_53
.LBB0_195:
	v_add_nc_u32_e32 v49, v57, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v50, 31, v49
	v_lshlrev_b64 v[49:50], 1, v[49:50]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v49, s3, s12, v49
	v_add_co_ci_u32_e64 v50, null, s13, v50, s3
	global_store_d16_hi_b16 v[49:50], v52, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_54
.LBB0_196:
	v_add_nc_u32_e32 v49, v57, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v50, 31, v49
	v_lshlrev_b64 v[49:50], 1, v[49:50]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v49, s3, s12, v49
	v_add_co_ci_u32_e64 v50, null, s13, v50, s3
	global_store_d16_hi_b16 v[49:50], v53, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_55
.LBB0_197:
	v_add_nc_u32_e32 v49, v57, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v50, 31, v49
	v_lshlrev_b64 v[49:50], 1, v[49:50]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v49, s3, s12, v49
	v_add_co_ci_u32_e64 v50, null, s13, v50, s3
	global_store_d16_hi_b16 v[49:50], v54, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_56
.LBB0_198:
	v_add_nc_u32_e32 v49, v57, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v50, 31, v49
	v_lshlrev_b64 v[49:50], 1, v[49:50]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v49, s3, s12, v49
	v_add_co_ci_u32_e64 v50, null, s13, v50, s3
	global_store_d16_hi_b16 v[49:50], v55, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_57
	s_branch .LBB0_58
.LBB0_199:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v58, v0, v52
	v_ashrrev_i32_e32 v59, 31, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	v_add_co_u32 v58, s11, s12, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v59, null, s13, v59, s11
	global_store_b16 v[58:59], v48, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s0
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_74
.LBB0_200:
	v_add_nc_u32_e32 v58, v66, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s11
	global_store_b16 v[58:59], v33, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_75
.LBB0_201:
	v_add_nc_u32_e32 v58, v66, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s11
	global_store_b16 v[58:59], v34, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_76
.LBB0_202:
	v_add_nc_u32_e32 v58, v66, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s11
	global_store_b16 v[58:59], v35, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_77
.LBB0_203:
	v_add_nc_u32_e32 v58, v66, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s11
	global_store_b16 v[58:59], v36, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_78
.LBB0_204:
	v_add_nc_u32_e32 v58, v66, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s11
	global_store_b16 v[58:59], v37, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_79
.LBB0_205:
	v_add_nc_u32_e32 v58, v66, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s11
	global_store_b16 v[58:59], v38, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_80
.LBB0_206:
	v_add_nc_u32_e32 v58, v66, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s11
	global_store_b16 v[58:59], v39, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_81
.LBB0_207:
	v_add_nc_u32_e32 v58, v66, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s11
	global_store_b16 v[58:59], v40, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_82
.LBB0_208:
	v_add_nc_u32_e32 v58, v67, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s11
	global_store_d16_hi_b16 v[58:59], v41, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_83
.LBB0_209:
	v_add_nc_u32_e32 v58, v67, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s12, v58
	v_add_co_ci_u32_e64 v59, null, s13, v59, s11
	global_store_d16_hi_b16 v[58:59], v42, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_84
.LBB0_210:
	v_add_nc_u32_e32 v41, v67, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_d16_hi_b16 v[41:42], v43, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_85
.LBB0_211:
	v_add_nc_u32_e32 v41, v67, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_d16_hi_b16 v[41:42], v44, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_86
.LBB0_212:
	v_add_nc_u32_e32 v41, v67, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_d16_hi_b16 v[41:42], v45, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_87
.LBB0_213:
	v_add_nc_u32_e32 v41, v67, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_d16_hi_b16 v[41:42], v46, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_88
.LBB0_214:
	v_add_nc_u32_e32 v41, v67, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_d16_hi_b16 v[41:42], v47, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_89
.LBB0_215:
	v_add_nc_u32_e32 v41, v67, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_d16_hi_b16 v[41:42], v48, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s3, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s3
	s_cbranch_execz .LBB0_90
.LBB0_216:
	v_add_nc_u32_e32 v41, v57, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s3, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s3
	global_store_d16_hi_b16 v[41:42], v33, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s3, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_91
.LBB0_217:
	v_add_nc_u32_e32 v41, v57, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s3, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s3
	global_store_d16_hi_b16 v[41:42], v34, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_92
.LBB0_218:
	v_add_nc_u32_e32 v33, v57, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v34, 31, v33
	v_lshlrev_b64 v[33:34], 1, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s3, s12, v33
	v_add_co_ci_u32_e64 v34, null, s13, v34, s3
	global_store_d16_hi_b16 v[33:34], v35, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_93
.LBB0_219:
	v_add_nc_u32_e32 v33, v57, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v34, 31, v33
	v_lshlrev_b64 v[33:34], 1, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s3, s12, v33
	v_add_co_ci_u32_e64 v34, null, s13, v34, s3
	global_store_d16_hi_b16 v[33:34], v36, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_94
.LBB0_220:
	v_add_nc_u32_e32 v33, v57, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v34, 31, v33
	v_lshlrev_b64 v[33:34], 1, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s3, s12, v33
	v_add_co_ci_u32_e64 v34, null, s13, v34, s3
	global_store_d16_hi_b16 v[33:34], v37, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_95
.LBB0_221:
	v_add_nc_u32_e32 v33, v57, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v34, 31, v33
	v_lshlrev_b64 v[33:34], 1, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s3, s12, v33
	v_add_co_ci_u32_e64 v34, null, s13, v34, s3
	global_store_d16_hi_b16 v[33:34], v38, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_96
.LBB0_222:
	v_add_nc_u32_e32 v33, v57, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v34, 31, v33
	v_lshlrev_b64 v[33:34], 1, v[33:34]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s3, s12, v33
	v_add_co_ci_u32_e64 v34, null, s13, v34, s3
	global_store_d16_hi_b16 v[33:34], v39, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_97
	s_branch .LBB0_98
.LBB0_223:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v41, v0, v36
	v_ashrrev_i32_e32 v42, 31, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	v_add_co_u32 v41, s11, s12, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_b16 v[41:42], v32, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s0
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_114
.LBB0_224:
	v_add_nc_u32_e32 v41, v66, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_b16 v[41:42], v17, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_115
.LBB0_225:
	v_add_nc_u32_e32 v41, v66, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_b16 v[41:42], v18, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_116
.LBB0_226:
	v_add_nc_u32_e32 v41, v66, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_b16 v[41:42], v19, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_117
.LBB0_227:
	v_add_nc_u32_e32 v41, v66, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_b16 v[41:42], v20, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_118
.LBB0_228:
	v_add_nc_u32_e32 v41, v66, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_b16 v[41:42], v21, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_119
.LBB0_229:
	v_add_nc_u32_e32 v41, v66, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_b16 v[41:42], v22, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_120
.LBB0_230:
	v_add_nc_u32_e32 v41, v66, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_b16 v[41:42], v23, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_121
.LBB0_231:
	v_add_nc_u32_e32 v41, v66, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_b16 v[41:42], v24, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_122
.LBB0_232:
	v_add_nc_u32_e32 v41, v67, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_d16_hi_b16 v[41:42], v25, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_123
.LBB0_233:
	v_add_nc_u32_e32 v41, v67, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s12, v41
	v_add_co_ci_u32_e64 v42, null, s13, v42, s11
	global_store_d16_hi_b16 v[41:42], v26, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_124
.LBB0_234:
	v_add_nc_u32_e32 v25, v67, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s11, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, s11
	global_store_d16_hi_b16 v[25:26], v27, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_125
.LBB0_235:
	v_add_nc_u32_e32 v25, v67, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s11, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, s11
	global_store_d16_hi_b16 v[25:26], v28, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_126
.LBB0_236:
	v_add_nc_u32_e32 v25, v67, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s11, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, s11
	global_store_d16_hi_b16 v[25:26], v29, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_127
.LBB0_237:
	v_add_nc_u32_e32 v25, v67, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s11, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, s11
	global_store_d16_hi_b16 v[25:26], v30, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_128
.LBB0_238:
	v_add_nc_u32_e32 v25, v67, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s11, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, s11
	global_store_d16_hi_b16 v[25:26], v31, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_129
.LBB0_239:
	v_add_nc_u32_e32 v25, v67, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s11, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, s11
	global_store_d16_hi_b16 v[25:26], v32, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s3, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s3
	s_cbranch_execz .LBB0_130
.LBB0_240:
	v_add_nc_u32_e32 v25, v57, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s3, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, s3
	global_store_d16_hi_b16 v[25:26], v17, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s3, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_131
.LBB0_241:
	v_add_nc_u32_e32 v25, v57, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s3, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, s3
	global_store_d16_hi_b16 v[25:26], v18, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_132
.LBB0_242:
	v_add_nc_u32_e32 v17, v57, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v17, s3, s12, v17
	v_add_co_ci_u32_e64 v18, null, s13, v18, s3
	global_store_d16_hi_b16 v[17:18], v19, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_133
.LBB0_243:
	v_add_nc_u32_e32 v17, v57, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v17, s3, s12, v17
	v_add_co_ci_u32_e64 v18, null, s13, v18, s3
	global_store_d16_hi_b16 v[17:18], v20, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_134
.LBB0_244:
	v_add_nc_u32_e32 v17, v57, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v17, s3, s12, v17
	v_add_co_ci_u32_e64 v18, null, s13, v18, s3
	global_store_d16_hi_b16 v[17:18], v21, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_135
.LBB0_245:
	v_add_nc_u32_e32 v17, v57, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v17, s3, s12, v17
	v_add_co_ci_u32_e64 v18, null, s13, v18, s3
	global_store_d16_hi_b16 v[17:18], v22, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_136
.LBB0_246:
	v_add_nc_u32_e32 v17, v57, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v17, s3, s12, v17
	v_add_co_ci_u32_e64 v18, null, s13, v18, s3
	global_store_d16_hi_b16 v[17:18], v23, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_137
	s_branch .LBB0_138
.LBB0_247:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v25, v0, v24
	v_ashrrev_i32_e32 v26, 31, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	v_add_co_u32 v25, vcc_lo, s12, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v26, null, s13, v26, vcc_lo
	global_store_b16 v[25:26], v16, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s3, s0
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB0_154
.LBB0_248:
	v_add_nc_u32_e32 v25, v66, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, vcc_lo, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, vcc_lo
	global_store_b16 v[25:26], v1, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB0_155
.LBB0_249:
	v_add_nc_u32_e32 v25, v66, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, vcc_lo, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, vcc_lo
	global_store_b16 v[25:26], v2, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB0_156
.LBB0_250:
	v_add_nc_u32_e32 v25, v66, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, vcc_lo, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, vcc_lo
	global_store_b16 v[25:26], v3, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB0_157
.LBB0_251:
	v_add_nc_u32_e32 v25, v66, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, vcc_lo, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, vcc_lo
	global_store_b16 v[25:26], v4, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB0_158
.LBB0_252:
	v_add_nc_u32_e32 v25, v66, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, vcc_lo, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, vcc_lo
	global_store_b16 v[25:26], v5, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB0_159
.LBB0_253:
	v_add_nc_u32_e32 v25, v66, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, vcc_lo, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, vcc_lo
	global_store_b16 v[25:26], v6, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s14, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB0_160
.LBB0_254:
	v_add_nc_u32_e32 v25, v66, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, vcc_lo, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, vcc_lo
	global_store_b16 v[25:26], v7, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_161
.LBB0_255:
	v_add_nc_u32_e32 v25, v66, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, vcc_lo, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, vcc_lo
	global_store_b16 v[25:26], v8, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_162
.LBB0_256:
	v_add_nc_u32_e32 v25, v67, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, vcc_lo, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, vcc_lo
	global_store_d16_hi_b16 v[25:26], v9, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_163
.LBB0_257:
	v_add_nc_u32_e32 v25, v67, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, vcc_lo, s12, v25
	v_add_co_ci_u32_e64 v26, null, s13, v26, vcc_lo
	global_store_d16_hi_b16 v[25:26], v10, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_164
.LBB0_258:
	v_add_nc_u32_e32 v9, v67, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v9, vcc_lo, s12, v9
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v11, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_165
.LBB0_259:
	v_add_nc_u32_e32 v9, v67, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v9, vcc_lo, s12, v9
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v12, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_166
.LBB0_260:
	v_add_nc_u32_e32 v9, v67, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v9, vcc_lo, s12, v9
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v13, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_167
.LBB0_261:
	v_add_nc_u32_e32 v9, v67, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v9, vcc_lo, s12, v9
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v14, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_168
.LBB0_262:
	v_add_nc_u32_e32 v9, v67, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v9, vcc_lo, s12, v9
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v15, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_169
.LBB0_263:
	v_add_nc_u32_e32 v9, v67, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v9, vcc_lo, s12, v9
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v16, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_170
.LBB0_264:
	v_add_nc_u32_e32 v9, v57, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v9, vcc_lo, s12, v9
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v1, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_171
.LBB0_265:
	v_add_nc_u32_e32 v0, v57, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s12, v0
	v_add_co_ci_u32_e64 v1, null, s13, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v2, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_172
.LBB0_266:
	v_add_nc_u32_e32 v0, v57, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s12, v0
	v_add_co_ci_u32_e64 v1, null, s13, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v3, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_173
.LBB0_267:
	v_add_nc_u32_e32 v0, v57, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s12, v0
	v_add_co_ci_u32_e64 v1, null, s13, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v4, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_174
.LBB0_268:
	v_add_nc_u32_e32 v0, v57, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s12, v0
	v_add_co_ci_u32_e64 v1, null, s13, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v5, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_175
.LBB0_269:
	v_add_nc_u32_e32 v0, v57, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s12, v0
	v_add_co_ci_u32_e64 v1, null, s13, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v6, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_176
.LBB0_270:
	v_add_nc_u32_e32 v0, v57, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s12, v0
	v_add_co_ci_u32_e64 v1, null, s13, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v7, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s0, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_177
.LBB0_271:
	v_add_nc_u32_e32 v0, v57, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s12, v0
	v_add_co_ci_u32_e64 v1, null, s13, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v8, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii, .Lfunc_end0-_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii
		.amdhsa_group_segment_fixed_size 12288
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 36
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 123
		.amdhsa_next_free_sgpr 20
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size 63
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii,"axG",@progbits,_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii,comdat
                                        ; -- End function
	.set .L_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii.num_vgpr, 118
	.set .L_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii.num_agpr, 0
	.set .L_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii.numbered_sgpr, 20
	.set .L_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii.num_named_barrier, 0
	.set .L_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii.private_seg_size, 0
	.set .L_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii.uses_vcc, 1
	.set .L_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii.uses_flat_scratch, 0
	.set .L_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii.has_dyn_sized_stack, 0
	.set .L_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii.has_recursion, 0
	.set .L_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 13340
; TotalNumSgprs: 22
; NumVgprs: 118
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 12288 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 118
; Occupancy: 12
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_56d8bc841b74ba2c,@object ; @__hip_cuid_56d8bc841b74ba2c
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_56d8bc841b74ba2c
__hip_cuid_56d8bc841b74ba2c:
	.byte	0                               ; 0x0
	.size	__hip_cuid_56d8bc841b74ba2c, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_56d8bc841b74ba2c
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  write_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 12288
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     123
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 0
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
