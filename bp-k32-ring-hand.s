	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 6
	.section	.text._ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii,"axG",@progbits,_ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii,comdat
	.protected	_ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii ; -- Begin function _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii
	.globl	_ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii
	.p2align	8
	.type	_ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii,@function
_ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii: ; @_ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii
; %bb.0:
	s_load_b128 s[12:15], s[0:1], 0x18
	s_abs_i32 s6, s2
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s3, s12, 31
	s_ashr_i32 s9, s13, 31
	s_lshr_b32 s3, s3, 24
	s_lshr_b32 s8, s9, 25
	s_add_i32 s3, s12, s3
	s_add_i32 s8, s13, s8
	s_ashr_i32 s3, s3, 8
	s_ashr_i32 s8, s8, 7
	s_abs_i32 s4, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_f32_u32 s5, s4
	s_sub_i32 s7, 0, s4
	v_rcp_iflag_f32_e32 v1, s5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s5, v1
	s_mul_f32 s5, s5, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s5, s5
	s_mul_i32 s7, s7, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s7, s5, s7
	s_add_i32 s5, s5, s7
	s_xor_b32 s7, s2, s3
	s_mul_hi_u32 s5, s6, s5
	s_ashr_i32 s7, s7, 31
	s_mul_i32 s10, s5, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_i32 s6, s6, s10
	s_add_i32 s10, s5, 1
	s_sub_i32 s11, s6, s4
	s_cmp_ge_u32 s6, s4
	s_cselect_b32 s5, s10, s5
	s_cselect_b32 s6, s11, s6
	s_add_i32 s10, s5, 1
	s_cmp_ge_u32 s6, s4
	s_cselect_b32 s4, s10, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s5, s4, s7
	s_mov_b32 s4, 0
	s_sub_i32 s6, s5, s7
	s_lshr_b32 s5, s9, 21
	s_ashr_i32 s7, s6, 31
	s_add_i32 s9, s13, s5
	s_lshr_b32 s5, s7, 28
	s_mul_i32 s7, s6, s3
	s_add_i32 s10, s6, s5
	s_sub_i32 s5, s2, s7
	s_and_b32 s7, s10, -16
	s_ashr_i32 s2, s10, 4
	s_ashr_i32 s9, s9, 11
	s_sub_i32 s6, s6, s7
	s_cmp_ge_i32 s2, s9
	s_mul_i32 s6, s6, s3
	s_cbranch_scc0 .LBB0_2
; %bb.1:
	s_lshr_b32 s7, s8, 28
	s_add_i32 s15, s6, s5
	s_add_i32 s7, s8, s7
	s_abs_i32 s16, s15
	s_and_b32 s7, s7, -16
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_i32 s7, s8, s7
	s_abs_i32 s9, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_f32_u32 s10, s9
	s_sub_i32 s11, 0, s9
	v_rcp_iflag_f32_e32 v1, s10
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s10, v1
	s_mul_f32 s10, s10, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s10, s10
	s_mul_i32 s11, s11, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s11, s10, s11
	s_add_i32 s10, s10, s11
	s_xor_b32 s11, s15, s7
	s_mul_hi_u32 s10, s16, s10
	s_ashr_i32 s11, s11, 31
	s_mul_i32 s17, s10, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_i32 s16, s16, s17
	s_add_i32 s17, s10, 1
	s_sub_i32 s18, s16, s9
	s_cmp_ge_u32 s16, s9
	s_cselect_b32 s10, s17, s10
	s_cselect_b32 s16, s18, s16
	s_add_i32 s17, s10, 1
	s_cmp_ge_u32 s16, s9
	s_cselect_b32 s9, s17, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s9, s9, s11
	s_sub_i32 s9, s9, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s7, s9, s7
	s_sub_i32 s10, s15, s7
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccz .LBB0_3
	s_branch .LBB0_4
.LBB0_2:
                                        ; implicit-def: $sgpr9
                                        ; implicit-def: $sgpr10
.LBB0_3:
	s_add_i32 s4, s6, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s5, s4, 31
	s_lshr_b32 s5, s5, 28
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_i32 s5, s4, s5
	s_and_b32 s6, s5, -16
	s_ashr_i32 s9, s5, 4
	s_sub_i32 s4, s4, s6
	s_and_b32 s5, s9, 15
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s10, s5, s4
.LBB0_4:
	s_load_b128 s[4:7], s[0:1], 0x8
	s_lshl_b32 s11, s2, 4
	s_and_b32 s2, s2, 1
	s_add_i32 s10, s10, s11
	s_not_b32 s11, s9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_add_i32 s11, s3, s11
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, s9, s11
	s_cmp_lt_i32 s2, s3
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s10, s8
	s_cselect_b32 s11, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s9, s9, s11
	s_mov_b32 s11, 0
	s_and_b32 vcc_lo, exec_lo, s9
	s_cbranch_vccnz .LBB0_6
; %bb.5:
	s_lshl_b32 s3, s3, 8
	s_lshl_b32 s9, s8, 7
	s_add_i32 s8, s3, 0xffffff00
	s_addk_i32 s9, 0xff80
	s_and_not1_b32 vcc_lo, exec_lo, s11
	s_cbranch_vccz .LBB0_7
	s_branch .LBB0_8
.LBB0_6:
                                        ; implicit-def: $sgpr9
                                        ; implicit-def: $sgpr8
.LBB0_7:
	s_lshl_b32 s8, s2, 8
	s_lshl_b32 s9, s10, 7
.LBB0_8:                                ; %_ZNK14rocm_wmma_gemm11tile_mapperILi256ELi128ELNS_8m_layoutE1ELS1_0ELi16EE8map_tileEiiiPiS3_.exit
	s_ashr_i32 s11, s14, 31
	s_ashr_i32 s2, s8, 31
	s_lshr_b32 s3, s11, 27
	s_lshr_b32 s2, s2, 24
	s_add_i32 s3, s14, s3
	s_add_i32 s2, s8, s2
	s_ashr_i32 s10, s3, 5
	s_ashr_i32 s3, s9, 31
	s_ashr_i32 s2, s2, 8
	s_lshr_b32 s15, s3, 25
	s_mul_hi_i32 s3, s10, s2
	s_add_i32 s15, s9, s15
	s_mul_i32 s2, s10, s2
	s_ashr_i32 s15, s15, 7
	s_lshl_b64 s[2:3], s[2:3], 14
	v_lshlrev_b32_e32 v1, 5, v0
	s_mul_hi_i32 s17, s10, s15
	s_mul_i32 s16, s10, s15
	s_waitcnt lgkmcnt(0)
	s_add_u32 s2, s4, s2
	s_addc_u32 s3, s5, s3
	s_lshl_b64 s[4:5], s[16:17], 13
	v_lshlrev_b32_e32 v118, 4, v0
	s_add_u32 s4, s6, s4
	v_add_co_u32 v2, s6, s2, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, s3, 0, s6
	s_addc_u32 s5, s7, s5
	v_add_co_u32 v12, s6, s4, v118
	s_clause 0x1
	global_load_b128 v[3:6], v1, s[2:3]
	global_load_b128 v[7:10], v1, s[2:3] offset:16
	v_add_co_ci_u32_e64 v13, null, s5, 0, s6
	v_add_co_u32 v1, vcc_lo, 0x2000, v2
	v_add_co_ci_u32_e32 v2, vcc_lo, 0, v11, vcc_lo
	v_add_co_u32 v15, vcc_lo, 0x1000, v12
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_ci_u32_e32 v16, vcc_lo, 0, v13, vcc_lo
	global_load_b128 v[11:14], v118, s[4:5]
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_clause 0x1
	global_load_b128 v[105:108], v[1:2], off
	global_load_b128 v[109:112], v[1:2], off offset:16
	global_load_b128 v[113:116], v[15:16], off
	v_lshlrev_b32_e32 v65, 1, v0
	v_lshrrev_b32_e32 v15, 1, v0
	v_and_b32_e32 v16, 16, v118
	v_and_b32_e32 v1, 15, v0
	v_mul_u32_u24_e32 v119, 0x50, v0
	v_and_b32_e32 v2, 64, v65
	s_mov_b32 s6, 0
	v_mad_u32_u24 v67, 0x50, v15, v16
	s_cmp_gt_i32 s14, 15
	s_waitcnt vmcnt(5)
	ds_store_b128 v119, v[3:6]
	s_waitcnt vmcnt(4)
	ds_store_b128 v119, v[7:10] offset:16
	s_waitcnt vmcnt(3)
	ds_store_b128 v67, v[11:14] offset:20480
	v_or_b32_e32 v117, v2, v1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(2)
	ds_store_b128 v119, v[105:108] offset:32
	s_waitcnt vmcnt(1)
	ds_store_b128 v119, v[109:112] offset:48
	s_waitcnt vmcnt(0)
	ds_store_b128 v67, v[113:116] offset:20512
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_10
; %bb.9:                                ; %_ZNK14rocm_wmma_gemm11tile_mapperILi256ELi128ELNS_8m_layoutE1ELS1_0ELi16EE8map_tileEiiiPiS3_.exit..preheader195_crit_edge
	v_or_b32_e32 v66, v2, v1
	s_branch .LBB0_11
.LBB0_10:
	s_mov_b32 s6, -1
                                        ; implicit-def: $vgpr66
.LBB0_11:                               ; %Flow897
	s_load_b64 s[16:17], s[0:1], 0x0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v31, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v15, 0
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v13, 0
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v11, 0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v7, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v3, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_mov_b32 s10, 0
	s_cbranch_vccnz .LBB0_29
; %bb.12:                               ; %.lr.ph
	v_and_b32_e32 v1, 0xcf, v0
	s_lshr_b32 s0, s11, 28
	v_add_nc_u32_e32 v120, 0x5000, v67
	v_or_b32_e32 v2, 48, v0
	s_add_i32 s0, s14, s0
	v_mul_u32_u24_e32 v121, 0x50, v1
	v_mov_b32_e32 v1, 0
	s_ashr_i32 s11, s0, 4
	s_movk_i32 s0, 0x5000
	v_mul_u32_u24_e32 v122, 0x50, v2
	v_mad_u32_u24 v123, 0x50, v117, s0
	v_dual_mov_b32 v3, v1 :: v_dual_lshlrev_b32 v124, 4, v65
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v6, v1
	v_mov_b32_e32 v7, v1
	v_mov_b32_e32 v8, v1
	v_mov_b32_e32 v9, v1
	v_mov_b32_e32 v10, v1
	v_mov_b32_e32 v11, v1
	v_mov_b32_e32 v12, v1
	v_mov_b32_e32 v13, v1
	v_mov_b32_e32 v14, v1
	v_mov_b32_e32 v15, v1
	v_mov_b32_e32 v16, v1
	v_mov_b32_e32 v17, v1
	v_mov_b32_e32 v18, v1
	v_mov_b32_e32 v19, v1
	v_mov_b32_e32 v20, v1
	v_mov_b32_e32 v21, v1
	v_mov_b32_e32 v22, v1
	v_mov_b32_e32 v23, v1
	v_mov_b32_e32 v24, v1
	v_mov_b32_e32 v25, v1
	v_mov_b32_e32 v26, v1
	v_mov_b32_e32 v27, v1
	v_mov_b32_e32 v28, v1
	v_mov_b32_e32 v29, v1
	v_mov_b32_e32 v30, v1
	v_mov_b32_e32 v31, v1
	v_mov_b32_e32 v32, v1
	v_mov_b32_e32 v33, v1
	v_mov_b32_e32 v34, v1
	v_mov_b32_e32 v35, v1
	v_mov_b32_e32 v36, v1
	v_mov_b32_e32 v37, v1
	v_mov_b32_e32 v38, v1
	v_mov_b32_e32 v39, v1
	v_mov_b32_e32 v40, v1
	v_mov_b32_e32 v41, v1
	v_mov_b32_e32 v42, v1
	v_mov_b32_e32 v43, v1
	v_mov_b32_e32 v44, v1
	v_mov_b32_e32 v45, v1
	v_mov_b32_e32 v46, v1
	v_mov_b32_e32 v47, v1
	v_mov_b32_e32 v48, v1
	v_mov_b32_e32 v49, v1
	v_mov_b32_e32 v50, v1
	v_mov_b32_e32 v51, v1
	v_mov_b32_e32 v52, v1
	v_mov_b32_e32 v53, v1
	v_mov_b32_e32 v54, v1
	v_mov_b32_e32 v55, v1
	v_mov_b32_e32 v56, v1
	v_mov_b32_e32 v57, v1
	v_mov_b32_e32 v58, v1
	v_mov_b32_e32 v59, v1
	v_mov_b32_e32 v60, v1
	v_mov_b32_e32 v61, v1
	v_mov_b32_e32 v62, v1
	v_mov_b32_e32 v63, v1
	v_mov_b32_e32 v64, v1
	s_add_i32 s14, s11, -2
	s_mov_b32 s15, 0
	s_branch .LBB0_14
.LBB0_13:                               ;   in Loop: Header=BB0_14 Depth=1
	ds_load_b128 v[97:100], v127
	ds_load_b128 v[101:104], v127 offset:16
	s_waitcnt lgkmcnt(0)
	v_wmma_f16_16x16x16_f16 v[57:64], v[89:96], v[97:104], v[57:64]
	v_wmma_f16_16x16x16_f16 v[41:48], v[81:88], v[97:104], v[41:48]
	v_wmma_f16_16x16x16_f16 v[25:32], v[73:80], v[97:104], v[25:32]
	v_wmma_f16_16x16x16_f16 v[9:16], v[65:72], v[97:104], v[9:16]
	ds_load_b128 v[97:100], v127 offset:1280
	ds_load_b128 v[101:104], v127 offset:1296
	s_waitcnt lgkmcnt(0)
	v_wmma_f16_16x16x16_f16 v[49:56], v[89:96], v[97:104], v[49:56]
	v_wmma_f16_16x16x16_f16 v[33:40], v[81:88], v[97:104], v[33:40]
	v_wmma_f16_16x16x16_f16 v[17:24], v[73:80], v[97:104], v[17:24]
	v_wmma_f16_16x16x16_f16 v[1:8], v[65:72], v[97:104], v[1:8]
	ds_load_b128 v[97:100], v127 offset:2560
	ds_load_b128 v[101:104], v127 offset:2576
	s_waitcnt lgkmcnt(0)
	v_wmma_f16_16x16x16_f16 v[57:64], v[89:96], v[97:104], v[57:64] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[41:48], v[81:88], v[97:104], v[41:48] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[25:32], v[73:80], v[97:104], v[25:32] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[9:16], v[65:72], v[97:104], v[9:16] op_sel:[0,0,1]
	ds_load_b128 v[97:100], v127 offset:3840
	ds_load_b128 v[101:104], v127 offset:3856
	s_waitcnt lgkmcnt(0)
	v_wmma_f16_16x16x16_f16 v[49:56], v[89:96], v[97:104], v[49:56] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[33:40], v[81:88], v[97:104], v[33:40] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[17:24], v[73:80], v[97:104], v[17:24] op_sel:[0,0,1]
	v_wmma_f16_16x16x16_f16 v[1:8], v[65:72], v[97:104], v[1:8] op_sel:[0,0,1]
	s_add_i32 s15, s15, 1
	s_add_i32 s10, s10, 16
	s_cmp_eq_u32 s11, s15
	s_cbranch_scc1 .LBB0_28
.LBB0_14:                               ; =>This Inner Loop Header: Depth=1
	s_add_i32 s18, s15, 2
	s_mov_b64 s[0:1], 0
	s_cmp_lt_i32 s18, s11
	s_mov_b64 s[6:7], 0
	s_cselect_b32 s20, -1, 0
	s_cmp_ge_i32 s18, s11
	s_cselect_b32 s19, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s19
	s_cbranch_vccnz .LBB0_16
; %bb.15:                               ;   in Loop: Header=BB0_14 Depth=1
	s_lshr_b32 s0, s18, 1
	s_and_b32 s6, s15, 1
	s_lshl_b32 s1, s0, 14
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_u32 s1, s2, s1
	s_addc_u32 s7, s3, 0
	s_lshl_b32 s18, s6, 13
	s_lshl_b32 s21, s6, 12
	s_add_u32 s6, s1, s18
	s_addc_u32 s7, s7, 0
	s_lshl_b32 s0, s0, 13
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_u32 s0, s4, s0
	s_addc_u32 s1, s5, 0
	s_add_u32 s0, s0, s21
	s_addc_u32 s1, s1, 0
.LBB0_16:                               ; %.preheader197
                                        ;   in Loop: Header=BB0_14 Depth=1
	s_and_b32 s21, s10, 16
	v_add_co_u32 v97, s6, s6, v124
	s_lshl_b32 s18, s21, 1
	v_cndmask_b32_e64 v100, 0, 1, s20
	v_add_nc_u32_e32 v65, s18, v121
	v_add_nc_u32_e32 v69, s18, v122
	v_add_co_ci_u32_e64 v98, null, s7, 0, s6
	ds_load_b128 v[89:92], v65
	ds_load_b128 v[93:96], v65 offset:16
	ds_load_b128 v[85:88], v65 offset:1296
	ds_load_b128 v[81:84], v65 offset:1280
	ds_load_b128 v[77:80], v65 offset:2576
	ds_load_b128 v[73:76], v65 offset:2560
	ds_load_b128 v[65:68], v69
	ds_load_b128 v[69:72], v69 offset:16
	s_and_not1_b32 vcc_lo, exec_lo, s20
	s_cbranch_vccnz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_14 Depth=1
	s_waitcnt vmcnt(0)
	flat_load_b128 v[105:108], v[97:98]
.LBB0_18:                               ;   in Loop: Header=BB0_14 Depth=1
	v_lshl_add_u32 v127, s21, 1, v123
	v_cmp_ne_u32_e32 vcc_lo, 1, v100
	s_cbranch_vccnz .LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_14 Depth=1
	s_waitcnt vmcnt(0)
	flat_load_b128 v[109:112], v[97:98] offset:16
.LBB0_20:                               ;   in Loop: Header=BB0_14 Depth=1
	v_cmp_ne_u32_e32 vcc_lo, 1, v100
	s_cbranch_vccnz .LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_14 Depth=1
	v_add_co_u32 v97, s0, s0, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v98, null, s1, 0, s0
	s_waitcnt vmcnt(0)
	flat_load_b128 v[113:116], v[97:98]
.LBB0_22:                               ;   in Loop: Header=BB0_14 Depth=1
	s_and_b32 vcc_lo, exec_lo, s19
	s_mov_b32 s0, -1
	s_cbranch_vccz .LBB0_26
; %bb.23:                               ;   in Loop: Header=BB0_14 Depth=1
	s_cmp_lg_u32 s14, s15
	s_cbranch_scc1 .LBB0_25
; %bb.24:                               ;   in Loop: Header=BB0_14 Depth=1
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
.LBB0_25:                               ; %Flow
                                        ;   in Loop: Header=BB0_14 Depth=1
	s_mov_b32 s0, 0
.LBB0_26:                               ; %Flow895
                                        ;   in Loop: Header=BB0_14 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB0_13
; %bb.27:                               ;   in Loop: Header=BB0_14 Depth=1
	v_add_nc_u32_e32 v125, s18, v119
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_add_nc_u32_e32 v126, s18, v120
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_store_b128 v125, v[105:108]
	ds_store_b128 v125, v[109:112] offset:16
	ds_store_b128 v126, v[113:116]
	s_branch .LBB0_13
.LBB0_28:                               ; %.preheader195.loopexit
	v_mov_b32_e32 v66, v117
.LBB0_29:                               ; %Flow898
	v_and_b32_e32 v65, 0xc0, v0
	v_lshrrev_b32_e32 v0, 4, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_or_b32 v0, v0, 1, v65
	v_add_nc_u32_e32 v65, s8, v0
	v_add_nc_u32_e32 v0, s9, v66
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_lo_u32 v68, v65, s13
	v_cmp_gt_i32_e64 s3, s12, v65
	v_cmp_gt_i32_e32 vcc_lo, s13, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b32 s0, s3, vcc_lo
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_31
; %bb.30:
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v66, v0, v68
	v_ashrrev_i32_e32 v67, 31, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v66, s0, s16, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v67, s0, s17, v67, s0
	global_store_b16 v[66:67], v57, off
.LBB0_31:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 2, v65
	s_lshl_b32 s14, s13, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v69, s14, v68
	v_cmp_gt_i32_e64 s4, s12, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b32 s0, s4, vcc_lo
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_33
; %bb.32:
	v_add_nc_u32_e32 v66, v0, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s16, v66
	v_add_co_ci_u32_e64 v67, s0, s17, v67, s0
	global_store_b16 v[66:67], v58, off
.LBB0_33:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 4, v65
	v_add_nc_u32_e32 v70, s14, v69
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s12, v66
	s_and_b32 s0, s5, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_35
; %bb.34:
	v_add_nc_u32_e32 v66, v0, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s16, v66
	v_add_co_ci_u32_e64 v67, s0, s17, v67, s0
	global_store_b16 v[66:67], v59, off
.LBB0_35:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 6, v65
	v_add_nc_u32_e32 v71, s14, v70
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s12, v66
	s_and_b32 s0, s6, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_37
; %bb.36:
	v_add_nc_u32_e32 v66, v0, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s16, v66
	v_add_co_ci_u32_e64 v67, s0, s17, v67, s0
	global_store_b16 v[66:67], v60, off
.LBB0_37:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 8, v65
	v_add_nc_u32_e32 v72, s14, v71
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v66
	s_and_b32 s0, s7, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_39
; %bb.38:
	v_add_nc_u32_e32 v66, v0, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s16, v66
	v_add_co_ci_u32_e64 v67, s0, s17, v67, s0
	global_store_b16 v[66:67], v61, off
.LBB0_39:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 10, v65
	v_add_nc_u32_e32 v73, s14, v72
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v66
	s_and_b32 s0, s8, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_41
; %bb.40:
	v_add_nc_u32_e32 v66, v0, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s16, v66
	v_add_co_ci_u32_e64 v67, s0, s17, v67, s0
	global_store_b16 v[66:67], v62, off
.LBB0_41:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 12, v65
	v_add_nc_u32_e32 v74, s14, v73
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v66
	s_and_b32 s0, s9, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_43
; %bb.42:
	v_add_nc_u32_e32 v66, v0, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s16, v66
	v_add_co_ci_u32_e64 v67, s0, s17, v67, s0
	global_store_b16 v[66:67], v63, off
.LBB0_43:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 14, v65
	v_add_nc_u32_e32 v75, s14, v74
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v66
	s_and_b32 s0, s10, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_45
; %bb.44:
	v_add_nc_u32_e32 v66, v0, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshlrev_b64 v[66:67], 1, v[66:67]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v66, s0, s16, v66
	v_add_co_ci_u32_e64 v67, s0, s17, v67, s0
	global_store_b16 v[66:67], v64, off
.LBB0_45:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v66, 16, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s13, v66
	s_and_b32 s1, s3, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_192
; %bb.46:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_193
.LBB0_47:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_194
.LBB0_48:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_195
.LBB0_49:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_196
.LBB0_50:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_197
.LBB0_51:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_198
.LBB0_52:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.1
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_54
.LBB0_53:
	v_add_nc_u32_e32 v76, v66, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s16, v76
	v_add_co_ci_u32_e64 v77, s1, s17, v77, s1
	global_store_b16 v[76:77], v56, off
.LBB0_54:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.1
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v67, 32, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s13, v67
	s_and_b32 s2, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_199
; %bb.55:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_200
.LBB0_56:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_201
.LBB0_57:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_202
.LBB0_58:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_203
.LBB0_59:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_204
.LBB0_60:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_205
.LBB0_61:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_63
.LBB0_62:
	v_add_nc_u32_e32 v57, v67, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64 v[57:58], 1, v[57:58]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v57, s2, s16, v57
	v_add_co_ci_u32_e64 v58, s2, s17, v58, s2
	global_store_d16_hi_b16 v[57:58], v64, off
.LBB0_63:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.2
	s_or_b32 exec_lo, exec_lo, s11
	v_add_nc_u32_e32 v57, 48, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s13, v57
	s_and_b32 s3, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s3
	s_cbranch_execnz .LBB0_206
; %bb.64:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s3, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_207
.LBB0_65:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_208
.LBB0_66:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_209
.LBB0_67:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_210
.LBB0_68:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_211
.LBB0_69:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_212
.LBB0_70:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.3
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_72
.LBB0_71:
	v_add_nc_u32_e32 v49, v57, v75
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v50, 31, v49
	v_lshlrev_b64 v[49:50], 1, v[49:50]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v49, s3, s16, v49
	v_add_co_ci_u32_e64 v50, s3, s17, v50, s3
	global_store_d16_hi_b16 v[49:50], v56, off
.LBB0_72:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.3
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v50, 16, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v49, v50, s13
	v_cmp_gt_i32_e64 s3, s12, v50
	s_and_b32 s4, s3, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB0_74
; %bb.73:
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v50, v0, v49
	v_ashrrev_i32_e32 v51, 31, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[50:51], 1, v[50:51]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v50, s4, s16, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v51, s4, s17, v51, s4
	global_store_b16 v[50:51], v41, off
.LBB0_74:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.1242
	s_or_b32 exec_lo, exec_lo, s5
	v_add_nc_u32_e32 v50, 18, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s4, s12, v50
	v_add_nc_u32_e32 v50, s14, v49
	s_and_b32 s5, s4, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB0_76
; %bb.75:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v51, v0, v50
	v_ashrrev_i32_e32 v52, 31, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[51:52], 1, v[51:52]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v51, s5, s16, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v52, s5, s17, v52, s5
	global_store_b16 v[51:52], v42, off
.LBB0_76:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.1244
	s_or_b32 exec_lo, exec_lo, s6
	v_add_nc_u32_e32 v51, 20, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s5, s12, v51
	v_add_nc_u32_e32 v51, s14, v50
	s_and_b32 s6, s5, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_78
; %bb.77:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v52, v0, v51
	v_ashrrev_i32_e32 v53, 31, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[52:53], 1, v[52:53]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v52, s6, s16, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v53, s6, s17, v53, s6
	global_store_b16 v[52:53], v43, off
.LBB0_78:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.1246
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v52, 22, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s6, s12, v52
	v_add_nc_u32_e32 v52, s14, v51
	s_and_b32 s7, s6, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_80
; %bb.79:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v53, v0, v52
	v_ashrrev_i32_e32 v54, 31, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[53:54], 1, v[53:54]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v53, s7, s16, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v54, s7, s17, v54, s7
	global_store_b16 v[53:54], v44, off
.LBB0_80:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.1248
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v53, 24, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s7, s12, v53
	v_add_nc_u32_e32 v53, s14, v52
	s_and_b32 s8, s7, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB0_82
; %bb.81:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v54, v0, v53
	v_ashrrev_i32_e32 v55, 31, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[54:55], 1, v[54:55]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v54, s8, s16, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v55, s8, s17, v55, s8
	global_store_b16 v[54:55], v45, off
.LBB0_82:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.1250
	s_or_b32 exec_lo, exec_lo, s9
	v_add_nc_u32_e32 v54, 26, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s8, s12, v54
	v_add_nc_u32_e32 v54, s14, v53
	s_and_b32 s9, s8, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB0_84
; %bb.83:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v55, v0, v54
	v_ashrrev_i32_e32 v56, 31, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[55:56], 1, v[55:56]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v55, s9, s16, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v56, s9, s17, v56, s9
	global_store_b16 v[55:56], v46, off
.LBB0_84:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.1252
	s_or_b32 exec_lo, exec_lo, s10
	v_add_nc_u32_e32 v55, 28, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s9, s12, v55
	v_add_nc_u32_e32 v55, s14, v54
	s_and_b32 s10, s9, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB0_86
; %bb.85:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v58, v0, v55
	v_ashrrev_i32_e32 v59, 31, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v58, s10, s16, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v59, s10, s17, v59, s10
	global_store_b16 v[58:59], v47, off
.LBB0_86:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.1254
	s_or_b32 exec_lo, exec_lo, s11
	v_add_nc_u32_e32 v56, 30, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s10, s12, v56
	v_add_nc_u32_e32 v56, s14, v55
	s_and_b32 s11, s10, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_213
; %bb.87:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.1255
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_214
.LBB0_88:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_215
.LBB0_89:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_216
.LBB0_90:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_217
.LBB0_91:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_218
.LBB0_92:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_219
.LBB0_93:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_220
.LBB0_94:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_221
.LBB0_95:                               ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.1.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_222
.LBB0_96:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_223
.LBB0_97:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_224
.LBB0_98:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_225
.LBB0_99:                               ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_226
.LBB0_100:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_227
.LBB0_101:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_228
.LBB0_102:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_229
.LBB0_103:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.2.1
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s3, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s3
	s_cbranch_execnz .LBB0_230
.LBB0_104:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s3, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_231
.LBB0_105:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_232
.LBB0_106:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_233
.LBB0_107:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_234
.LBB0_108:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_235
.LBB0_109:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_236
.LBB0_110:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.3.1
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_112
.LBB0_111:
	v_add_nc_u32_e32 v33, v57, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v34, 31, v33
	v_lshlrev_b64 v[33:34], 1, v[33:34]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s3, s16, v33
	v_add_co_ci_u32_e64 v34, s3, s17, v34, s3
	global_store_d16_hi_b16 v[33:34], v40, off
.LBB0_112:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.3.1
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v34, 32, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v33, v34, s13
	v_cmp_gt_i32_e64 s3, s12, v34
	s_and_b32 s4, s3, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB0_114
; %bb.113:
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v34, v0, v33
	v_ashrrev_i32_e32 v35, 31, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[34:35], 1, v[34:35]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v34, s4, s16, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v35, s4, s17, v35, s4
	global_store_b16 v[34:35], v25, off
.LBB0_114:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.2274
	s_or_b32 exec_lo, exec_lo, s5
	v_add_nc_u32_e32 v34, 34, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s4, s12, v34
	v_add_nc_u32_e32 v34, s14, v33
	s_and_b32 s5, s4, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB0_116
; %bb.115:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v35, v0, v34
	v_ashrrev_i32_e32 v36, 31, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[35:36], 1, v[35:36]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v35, s5, s16, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v36, s5, s17, v36, s5
	global_store_b16 v[35:36], v26, off
.LBB0_116:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.2276
	s_or_b32 exec_lo, exec_lo, s6
	v_add_nc_u32_e32 v35, 36, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s5, s12, v35
	v_add_nc_u32_e32 v35, s14, v34
	s_and_b32 s6, s5, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_118
; %bb.117:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v36, v0, v35
	v_ashrrev_i32_e32 v37, 31, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[36:37], 1, v[36:37]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v36, s6, s16, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v37, s6, s17, v37, s6
	global_store_b16 v[36:37], v27, off
.LBB0_118:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.2278
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v36, 38, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s6, s12, v36
	v_add_nc_u32_e32 v36, s14, v35
	s_and_b32 s7, s6, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_120
; %bb.119:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v37, v0, v36
	v_ashrrev_i32_e32 v38, 31, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[37:38], 1, v[37:38]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v37, s7, s16, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v38, s7, s17, v38, s7
	global_store_b16 v[37:38], v28, off
.LBB0_120:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.2280
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v37, 40, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s7, s12, v37
	v_add_nc_u32_e32 v37, s14, v36
	s_and_b32 s8, s7, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB0_122
; %bb.121:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v38, v0, v37
	v_ashrrev_i32_e32 v39, 31, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[38:39], 1, v[38:39]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v38, s8, s16, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v39, s8, s17, v39, s8
	global_store_b16 v[38:39], v29, off
.LBB0_122:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.2282
	s_or_b32 exec_lo, exec_lo, s9
	v_add_nc_u32_e32 v38, 42, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s8, s12, v38
	v_add_nc_u32_e32 v38, s14, v37
	s_and_b32 s9, s8, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB0_124
; %bb.123:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v39, v0, v38
	v_ashrrev_i32_e32 v40, 31, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[39:40], 1, v[39:40]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v39, s9, s16, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v40, s9, s17, v40, s9
	global_store_b16 v[39:40], v30, off
.LBB0_124:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.2284
	s_or_b32 exec_lo, exec_lo, s10
	v_add_nc_u32_e32 v39, 44, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s9, s12, v39
	v_add_nc_u32_e32 v39, s14, v38
	s_and_b32 s10, s9, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB0_126
; %bb.125:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v40, v0, v39
	v_ashrrev_i32_e32 v41, 31, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[40:41], 1, v[40:41]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v40, s10, s16, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v41, s10, s17, v41, s10
	global_store_b16 v[40:41], v31, off
.LBB0_126:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.2286
	s_or_b32 exec_lo, exec_lo, s11
	v_add_nc_u32_e32 v40, 46, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s10, s12, v40
	v_add_nc_u32_e32 v40, s14, v39
	s_and_b32 s11, s10, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_237
; %bb.127:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.2287
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_238
.LBB0_128:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_239
.LBB0_129:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_240
.LBB0_130:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_241
.LBB0_131:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_242
.LBB0_132:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_243
.LBB0_133:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_244
.LBB0_134:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_245
.LBB0_135:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.1.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_246
.LBB0_136:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_247
.LBB0_137:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_248
.LBB0_138:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_249
.LBB0_139:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_250
.LBB0_140:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_251
.LBB0_141:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_252
.LBB0_142:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execnz .LBB0_253
.LBB0_143:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.2.2
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s3, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s3
	s_cbranch_execnz .LBB0_254
.LBB0_144:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s3, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_255
.LBB0_145:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_256
.LBB0_146:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_257
.LBB0_147:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_258
.LBB0_148:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_259
.LBB0_149:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_260
.LBB0_150:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.3.2
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_152
.LBB0_151:
	v_add_nc_u32_e32 v17, v57, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v17, s3, s16, v17
	v_add_co_ci_u32_e64 v18, s3, s17, v18, s3
	global_store_d16_hi_b16 v[17:18], v24, off
.LBB0_152:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.3.2
	s_or_b32 exec_lo, exec_lo, s4
	v_add_nc_u32_e32 v18, 48, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, v18, s13
	v_cmp_gt_i32_e64 s3, s12, v18
	s_and_b32 s4, s3, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB0_154
; %bb.153:
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v18, v0, v17
	v_ashrrev_i32_e32 v19, 31, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[18:19], 1, v[18:19]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v18, s4, s16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v19, s4, s17, v19, s4
	global_store_b16 v[18:19], v9, off
.LBB0_154:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.3306
	s_or_b32 exec_lo, exec_lo, s5
	v_add_nc_u32_e32 v18, 50, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s4, s12, v18
	v_add_nc_u32_e32 v18, s14, v17
	s_and_b32 s5, s4, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB0_156
; %bb.155:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v19, v0, v18
	v_ashrrev_i32_e32 v20, 31, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[19:20], 1, v[19:20]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v19, s5, s16, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v20, s5, s17, v20, s5
	global_store_b16 v[19:20], v10, off
.LBB0_156:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.3308
	s_or_b32 exec_lo, exec_lo, s6
	v_add_nc_u32_e32 v19, 52, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s5, s12, v19
	v_add_nc_u32_e32 v19, s14, v18
	s_and_b32 s6, s5, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s7, s6
	s_cbranch_execz .LBB0_158
; %bb.157:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v20, v0, v19
	v_ashrrev_i32_e32 v21, 31, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[20:21], 1, v[20:21]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v20, s6, s16, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v21, s6, s17, v21, s6
	global_store_b16 v[20:21], v11, off
.LBB0_158:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.3310
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v20, 54, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s6, s12, v20
	v_add_nc_u32_e32 v20, s14, v19
	s_and_b32 s7, s6, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB0_160
; %bb.159:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v21, v0, v20
	v_ashrrev_i32_e32 v22, 31, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[21:22], 1, v[21:22]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v21, s7, s16, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v22, s7, s17, v22, s7
	global_store_b16 v[21:22], v12, off
.LBB0_160:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.3312
	s_or_b32 exec_lo, exec_lo, s8
	v_add_nc_u32_e32 v21, 56, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s7, s12, v21
	v_add_nc_u32_e32 v21, s14, v20
	s_and_b32 s8, s7, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB0_162
; %bb.161:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v22, v0, v21
	v_ashrrev_i32_e32 v23, 31, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[22:23], 1, v[22:23]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v22, s8, s16, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v23, s8, s17, v23, s8
	global_store_b16 v[22:23], v13, off
.LBB0_162:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.3314
	s_or_b32 exec_lo, exec_lo, s9
	v_add_nc_u32_e32 v22, 58, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s8, s12, v22
	v_add_nc_u32_e32 v22, s14, v21
	s_and_b32 s9, s8, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB0_164
; %bb.163:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v23, v0, v22
	v_ashrrev_i32_e32 v24, 31, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[23:24], 1, v[23:24]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v23, s9, s16, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v24, s9, s17, v24, s9
	global_store_b16 v[23:24], v14, off
.LBB0_164:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.3316
	s_or_b32 exec_lo, exec_lo, s10
	v_add_nc_u32_e32 v23, 60, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s9, s12, v23
	v_add_nc_u32_e32 v23, s14, v22
	s_and_b32 s10, s9, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB0_166
; %bb.165:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v24, v0, v23
	v_ashrrev_i32_e32 v25, 31, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[24:25], 1, v[24:25]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v24, s10, s16, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v25, s10, s17, v25, s10
	global_store_b16 v[24:25], v15, off
.LBB0_166:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.3318
	s_or_b32 exec_lo, exec_lo, s11
	v_add_nc_u32_e32 v24, 62, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s10, s12, v24
	v_add_nc_u32_e32 v24, s14, v23
	s_and_b32 s12, s10, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execnz .LBB0_261
; %bb.167:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.3319
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s3, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execnz .LBB0_262
.LBB0_168:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execnz .LBB0_263
.LBB0_169:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execnz .LBB0_264
.LBB0_170:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execnz .LBB0_265
.LBB0_171:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execnz .LBB0_266
.LBB0_172:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execnz .LBB0_267
.LBB0_173:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execnz .LBB0_268
.LBB0_174:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.1.3
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_269
.LBB0_175:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.1.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_270
.LBB0_176:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_271
.LBB0_177:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_272
.LBB0_178:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_273
.LBB0_179:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_274
.LBB0_180:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_275
.LBB0_181:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB0_276
.LBB0_182:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_277
.LBB0_183:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.2.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_278
.LBB0_184:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm0EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_279
.LBB0_185:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm1EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_280
.LBB0_186:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm2EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_281
.LBB0_187:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm3EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_282
.LBB0_188:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm4EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_283
.LBB0_189:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm5EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB0_284
.LBB0_190:                              ; %_ZZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb1E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiiiENKUlTnmvE_clILm6EEEDav.exit.i.i.3.3
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s0, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execnz .LBB0_285
.LBB0_191:                              ; %_ZN14rocm_wmma_gemm12store_matrixILNS_8m_layoutE0ELb0ELb0E6__halfLi16EEENSt9enable_ifIXaaeqT_LS1_0EntT0_EvE4typeEPT2_RNS_8fragmentIS6_XT3_EEEiiii.exit.3.3
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.LBB0_192:
	v_add_nc_u32_e32 v76, v66, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s16, v76
	v_add_co_ci_u32_e64 v77, s1, s17, v77, s1
	global_store_b16 v[76:77], v49, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_47
.LBB0_193:
	v_add_nc_u32_e32 v76, v66, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s16, v76
	v_add_co_ci_u32_e64 v77, s1, s17, v77, s1
	global_store_b16 v[76:77], v50, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_48
.LBB0_194:
	v_add_nc_u32_e32 v76, v66, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s16, v76
	v_add_co_ci_u32_e64 v77, s1, s17, v77, s1
	global_store_b16 v[76:77], v51, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_49
.LBB0_195:
	v_add_nc_u32_e32 v76, v66, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s16, v76
	v_add_co_ci_u32_e64 v77, s1, s17, v77, s1
	global_store_b16 v[76:77], v52, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_50
.LBB0_196:
	v_add_nc_u32_e32 v76, v66, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s16, v76
	v_add_co_ci_u32_e64 v77, s1, s17, v77, s1
	global_store_b16 v[76:77], v53, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_51
.LBB0_197:
	v_add_nc_u32_e32 v76, v66, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s16, v76
	v_add_co_ci_u32_e64 v77, s1, s17, v77, s1
	global_store_b16 v[76:77], v54, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_52
.LBB0_198:
	v_add_nc_u32_e32 v76, v66, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s1, s16, v76
	v_add_co_ci_u32_e64 v77, s1, s17, v77, s1
	global_store_b16 v[76:77], v55, off
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execnz .LBB0_53
	s_branch .LBB0_54
.LBB0_199:
	v_add_nc_u32_e32 v76, v67, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s2, s16, v76
	v_add_co_ci_u32_e64 v77, s2, s17, v77, s2
	global_store_d16_hi_b16 v[76:77], v57, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_56
.LBB0_200:
	v_add_nc_u32_e32 v76, v67, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshlrev_b64 v[76:77], 1, v[76:77]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v76, s2, s16, v76
	v_add_co_ci_u32_e64 v77, s2, s17, v77, s2
	global_store_d16_hi_b16 v[76:77], v58, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_57
.LBB0_201:
	v_add_nc_u32_e32 v57, v67, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64 v[57:58], 1, v[57:58]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v57, s2, s16, v57
	v_add_co_ci_u32_e64 v58, s2, s17, v58, s2
	global_store_d16_hi_b16 v[57:58], v59, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_58
.LBB0_202:
	v_add_nc_u32_e32 v57, v67, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64 v[57:58], 1, v[57:58]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v57, s2, s16, v57
	v_add_co_ci_u32_e64 v58, s2, s17, v58, s2
	global_store_d16_hi_b16 v[57:58], v60, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_59
.LBB0_203:
	v_add_nc_u32_e32 v57, v67, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64 v[57:58], 1, v[57:58]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v57, s2, s16, v57
	v_add_co_ci_u32_e64 v58, s2, s17, v58, s2
	global_store_d16_hi_b16 v[57:58], v61, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_60
.LBB0_204:
	v_add_nc_u32_e32 v57, v67, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64 v[57:58], 1, v[57:58]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v57, s2, s16, v57
	v_add_co_ci_u32_e64 v58, s2, s17, v58, s2
	global_store_d16_hi_b16 v[57:58], v62, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execz .LBB0_61
.LBB0_205:
	v_add_nc_u32_e32 v57, v67, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64 v[57:58], 1, v[57:58]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v57, s2, s16, v57
	v_add_co_ci_u32_e64 v58, s2, s17, v58, s2
	global_store_d16_hi_b16 v[57:58], v63, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s2, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s2
	s_cbranch_execnz .LBB0_62
	s_branch .LBB0_63
.LBB0_206:
	v_add_nc_u32_e32 v58, v57, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s3, s16, v58
	v_add_co_ci_u32_e64 v59, s3, s17, v59, s3
	global_store_d16_hi_b16 v[58:59], v49, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s3, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_65
.LBB0_207:
	v_add_nc_u32_e32 v58, v57, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s3, s16, v58
	v_add_co_ci_u32_e64 v59, s3, s17, v59, s3
	global_store_d16_hi_b16 v[58:59], v50, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_66
.LBB0_208:
	v_add_nc_u32_e32 v49, v57, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v50, 31, v49
	v_lshlrev_b64 v[49:50], 1, v[49:50]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v49, s3, s16, v49
	v_add_co_ci_u32_e64 v50, s3, s17, v50, s3
	global_store_d16_hi_b16 v[49:50], v51, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_67
.LBB0_209:
	v_add_nc_u32_e32 v49, v57, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v50, 31, v49
	v_lshlrev_b64 v[49:50], 1, v[49:50]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v49, s3, s16, v49
	v_add_co_ci_u32_e64 v50, s3, s17, v50, s3
	global_store_d16_hi_b16 v[49:50], v52, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_68
.LBB0_210:
	v_add_nc_u32_e32 v49, v57, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v50, 31, v49
	v_lshlrev_b64 v[49:50], 1, v[49:50]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v49, s3, s16, v49
	v_add_co_ci_u32_e64 v50, s3, s17, v50, s3
	global_store_d16_hi_b16 v[49:50], v53, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_69
.LBB0_211:
	v_add_nc_u32_e32 v49, v57, v73
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v50, 31, v49
	v_lshlrev_b64 v[49:50], 1, v[49:50]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v49, s3, s16, v49
	v_add_co_ci_u32_e64 v50, s3, s17, v50, s3
	global_store_d16_hi_b16 v[49:50], v54, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_70
.LBB0_212:
	v_add_nc_u32_e32 v49, v57, v74
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v50, 31, v49
	v_lshlrev_b64 v[49:50], 1, v[49:50]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v49, s3, s16, v49
	v_add_co_ci_u32_e64 v50, s3, s17, v50, s3
	global_store_d16_hi_b16 v[49:50], v55, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_71
	s_branch .LBB0_72
.LBB0_213:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v58, v0, v56
	v_ashrrev_i32_e32 v59, 31, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v58, s11, s16, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v59, s11, s17, v59, s11
	global_store_b16 v[58:59], v48, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s0
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_88
.LBB0_214:
	v_add_nc_u32_e32 v58, v66, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s16, v58
	v_add_co_ci_u32_e64 v59, s11, s17, v59, s11
	global_store_b16 v[58:59], v33, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_89
.LBB0_215:
	v_add_nc_u32_e32 v58, v66, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s16, v58
	v_add_co_ci_u32_e64 v59, s11, s17, v59, s11
	global_store_b16 v[58:59], v34, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_90
.LBB0_216:
	v_add_nc_u32_e32 v58, v66, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s16, v58
	v_add_co_ci_u32_e64 v59, s11, s17, v59, s11
	global_store_b16 v[58:59], v35, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_91
.LBB0_217:
	v_add_nc_u32_e32 v58, v66, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s16, v58
	v_add_co_ci_u32_e64 v59, s11, s17, v59, s11
	global_store_b16 v[58:59], v36, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_92
.LBB0_218:
	v_add_nc_u32_e32 v58, v66, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s16, v58
	v_add_co_ci_u32_e64 v59, s11, s17, v59, s11
	global_store_b16 v[58:59], v37, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_93
.LBB0_219:
	v_add_nc_u32_e32 v58, v66, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s16, v58
	v_add_co_ci_u32_e64 v59, s11, s17, v59, s11
	global_store_b16 v[58:59], v38, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_94
.LBB0_220:
	v_add_nc_u32_e32 v58, v66, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s16, v58
	v_add_co_ci_u32_e64 v59, s11, s17, v59, s11
	global_store_b16 v[58:59], v39, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_95
.LBB0_221:
	v_add_nc_u32_e32 v58, v66, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s16, v58
	v_add_co_ci_u32_e64 v59, s11, s17, v59, s11
	global_store_b16 v[58:59], v40, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_96
.LBB0_222:
	v_add_nc_u32_e32 v58, v67, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s16, v58
	v_add_co_ci_u32_e64 v59, s11, s17, v59, s11
	global_store_d16_hi_b16 v[58:59], v41, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_97
.LBB0_223:
	v_add_nc_u32_e32 v58, v67, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v59, 31, v58
	v_lshlrev_b64 v[58:59], 1, v[58:59]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, s11, s16, v58
	v_add_co_ci_u32_e64 v59, s11, s17, v59, s11
	global_store_d16_hi_b16 v[58:59], v42, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_98
.LBB0_224:
	v_add_nc_u32_e32 v41, v67, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_d16_hi_b16 v[41:42], v43, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_99
.LBB0_225:
	v_add_nc_u32_e32 v41, v67, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_d16_hi_b16 v[41:42], v44, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_100
.LBB0_226:
	v_add_nc_u32_e32 v41, v67, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_d16_hi_b16 v[41:42], v45, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_101
.LBB0_227:
	v_add_nc_u32_e32 v41, v67, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_d16_hi_b16 v[41:42], v46, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_102
.LBB0_228:
	v_add_nc_u32_e32 v41, v67, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_d16_hi_b16 v[41:42], v47, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_103
.LBB0_229:
	v_add_nc_u32_e32 v41, v67, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_d16_hi_b16 v[41:42], v48, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s3, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s3
	s_cbranch_execz .LBB0_104
.LBB0_230:
	v_add_nc_u32_e32 v41, v57, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s3, s16, v41
	v_add_co_ci_u32_e64 v42, s3, s17, v42, s3
	global_store_d16_hi_b16 v[41:42], v33, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s3, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_105
.LBB0_231:
	v_add_nc_u32_e32 v41, v57, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s3, s16, v41
	v_add_co_ci_u32_e64 v42, s3, s17, v42, s3
	global_store_d16_hi_b16 v[41:42], v34, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_106
.LBB0_232:
	v_add_nc_u32_e32 v33, v57, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v34, 31, v33
	v_lshlrev_b64 v[33:34], 1, v[33:34]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s3, s16, v33
	v_add_co_ci_u32_e64 v34, s3, s17, v34, s3
	global_store_d16_hi_b16 v[33:34], v35, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_107
.LBB0_233:
	v_add_nc_u32_e32 v33, v57, v52
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v34, 31, v33
	v_lshlrev_b64 v[33:34], 1, v[33:34]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s3, s16, v33
	v_add_co_ci_u32_e64 v34, s3, s17, v34, s3
	global_store_d16_hi_b16 v[33:34], v36, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_108
.LBB0_234:
	v_add_nc_u32_e32 v33, v57, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v34, 31, v33
	v_lshlrev_b64 v[33:34], 1, v[33:34]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s3, s16, v33
	v_add_co_ci_u32_e64 v34, s3, s17, v34, s3
	global_store_d16_hi_b16 v[33:34], v37, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_109
.LBB0_235:
	v_add_nc_u32_e32 v33, v57, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v34, 31, v33
	v_lshlrev_b64 v[33:34], 1, v[33:34]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s3, s16, v33
	v_add_co_ci_u32_e64 v34, s3, s17, v34, s3
	global_store_d16_hi_b16 v[33:34], v38, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_110
.LBB0_236:
	v_add_nc_u32_e32 v33, v57, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v34, 31, v33
	v_lshlrev_b64 v[33:34], 1, v[33:34]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s3, s16, v33
	v_add_co_ci_u32_e64 v34, s3, s17, v34, s3
	global_store_d16_hi_b16 v[33:34], v39, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_111
	s_branch .LBB0_112
.LBB0_237:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v41, v0, v40
	v_ashrrev_i32_e32 v42, 31, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v41, s11, s16, v41
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_b16 v[41:42], v32, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s0
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_128
.LBB0_238:
	v_add_nc_u32_e32 v41, v66, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_b16 v[41:42], v17, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_129
.LBB0_239:
	v_add_nc_u32_e32 v41, v66, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_b16 v[41:42], v18, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_130
.LBB0_240:
	v_add_nc_u32_e32 v41, v66, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_b16 v[41:42], v19, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_131
.LBB0_241:
	v_add_nc_u32_e32 v41, v66, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_b16 v[41:42], v20, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_132
.LBB0_242:
	v_add_nc_u32_e32 v41, v66, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_b16 v[41:42], v21, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_133
.LBB0_243:
	v_add_nc_u32_e32 v41, v66, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_b16 v[41:42], v22, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_134
.LBB0_244:
	v_add_nc_u32_e32 v41, v66, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_b16 v[41:42], v23, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_135
.LBB0_245:
	v_add_nc_u32_e32 v41, v66, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_b16 v[41:42], v24, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_136
.LBB0_246:
	v_add_nc_u32_e32 v41, v67, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_d16_hi_b16 v[41:42], v25, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_137
.LBB0_247:
	v_add_nc_u32_e32 v41, v67, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v42, 31, v41
	v_lshlrev_b64 v[41:42], 1, v[41:42]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v41, s11, s16, v41
	v_add_co_ci_u32_e64 v42, s11, s17, v42, s11
	global_store_d16_hi_b16 v[41:42], v26, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_138
.LBB0_248:
	v_add_nc_u32_e32 v25, v67, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s11, s16, v25
	v_add_co_ci_u32_e64 v26, s11, s17, v26, s11
	global_store_d16_hi_b16 v[25:26], v27, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_139
.LBB0_249:
	v_add_nc_u32_e32 v25, v67, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s11, s16, v25
	v_add_co_ci_u32_e64 v26, s11, s17, v26, s11
	global_store_d16_hi_b16 v[25:26], v28, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_140
.LBB0_250:
	v_add_nc_u32_e32 v25, v67, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s11, s16, v25
	v_add_co_ci_u32_e64 v26, s11, s17, v26, s11
	global_store_d16_hi_b16 v[25:26], v29, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_141
.LBB0_251:
	v_add_nc_u32_e32 v25, v67, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s11, s16, v25
	v_add_co_ci_u32_e64 v26, s11, s17, v26, s11
	global_store_d16_hi_b16 v[25:26], v30, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_142
.LBB0_252:
	v_add_nc_u32_e32 v25, v67, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s11, s16, v25
	v_add_co_ci_u32_e64 v26, s11, s17, v26, s11
	global_store_d16_hi_b16 v[25:26], v31, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s11, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s15, s11
	s_cbranch_execz .LBB0_143
.LBB0_253:
	v_add_nc_u32_e32 v25, v67, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s11, s16, v25
	v_add_co_ci_u32_e64 v26, s11, s17, v26, s11
	global_store_d16_hi_b16 v[25:26], v32, off
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s3, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s3
	s_cbranch_execz .LBB0_144
.LBB0_254:
	v_add_nc_u32_e32 v25, v57, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s3, s16, v25
	v_add_co_ci_u32_e64 v26, s3, s17, v26, s3
	global_store_d16_hi_b16 v[25:26], v17, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s3, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_145
.LBB0_255:
	v_add_nc_u32_e32 v25, v57, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s3, s16, v25
	v_add_co_ci_u32_e64 v26, s3, s17, v26, s3
	global_store_d16_hi_b16 v[25:26], v18, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_146
.LBB0_256:
	v_add_nc_u32_e32 v17, v57, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v17, s3, s16, v17
	v_add_co_ci_u32_e64 v18, s3, s17, v18, s3
	global_store_d16_hi_b16 v[17:18], v19, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_147
.LBB0_257:
	v_add_nc_u32_e32 v17, v57, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v17, s3, s16, v17
	v_add_co_ci_u32_e64 v18, s3, s17, v18, s3
	global_store_d16_hi_b16 v[17:18], v20, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_148
.LBB0_258:
	v_add_nc_u32_e32 v17, v57, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v17, s3, s16, v17
	v_add_co_ci_u32_e64 v18, s3, s17, v18, s3
	global_store_d16_hi_b16 v[17:18], v21, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_149
.LBB0_259:
	v_add_nc_u32_e32 v17, v57, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v17, s3, s16, v17
	v_add_co_ci_u32_e64 v18, s3, s17, v18, s3
	global_store_d16_hi_b16 v[17:18], v22, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB0_150
.LBB0_260:
	v_add_nc_u32_e32 v17, v57, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v18, 31, v17
	v_lshlrev_b64 v[17:18], 1, v[17:18]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v17, s3, s16, v17
	v_add_co_ci_u32_e64 v18, s3, s17, v18, s3
	global_store_d16_hi_b16 v[17:18], v23, off
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB0_151
	s_branch .LBB0_152
.LBB0_261:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v25, v0, v24
	v_ashrrev_i32_e32 v26, 31, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v25, vcc_lo, s16, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e32 v26, vcc_lo, s17, v26, vcc_lo
	global_store_b16 v[25:26], v16, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s3, s0
	s_and_saveexec_b32 s11, s12
	s_cbranch_execz .LBB0_168
.LBB0_262:
	v_add_nc_u32_e32 v25, v66, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s16, v25
	v_add_co_ci_u32_e32 v26, vcc_lo, s17, v26, vcc_lo
	global_store_b16 v[25:26], v1, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s4, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execz .LBB0_169
.LBB0_263:
	v_add_nc_u32_e32 v25, v66, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s16, v25
	v_add_co_ci_u32_e32 v26, vcc_lo, s17, v26, vcc_lo
	global_store_b16 v[25:26], v2, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s5, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execz .LBB0_170
.LBB0_264:
	v_add_nc_u32_e32 v25, v66, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s16, v25
	v_add_co_ci_u32_e32 v26, vcc_lo, s17, v26, vcc_lo
	global_store_b16 v[25:26], v3, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s6, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execz .LBB0_171
.LBB0_265:
	v_add_nc_u32_e32 v25, v66, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s16, v25
	v_add_co_ci_u32_e32 v26, vcc_lo, s17, v26, vcc_lo
	global_store_b16 v[25:26], v4, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s7, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execz .LBB0_172
.LBB0_266:
	v_add_nc_u32_e32 v25, v66, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s16, v25
	v_add_co_ci_u32_e32 v26, vcc_lo, s17, v26, vcc_lo
	global_store_b16 v[25:26], v5, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s8, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execz .LBB0_173
.LBB0_267:
	v_add_nc_u32_e32 v25, v66, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s16, v25
	v_add_co_ci_u32_e32 v26, vcc_lo, s17, v26, vcc_lo
	global_store_b16 v[25:26], v6, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s12, s9, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s11, s12
	s_cbranch_execz .LBB0_174
.LBB0_268:
	v_add_nc_u32_e32 v25, v66, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s16, v25
	v_add_co_ci_u32_e32 v26, vcc_lo, s17, v26, vcc_lo
	global_store_b16 v[25:26], v7, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_175
.LBB0_269:
	v_add_nc_u32_e32 v25, v66, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s16, v25
	v_add_co_ci_u32_e32 v26, vcc_lo, s17, v26, vcc_lo
	global_store_b16 v[25:26], v8, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s3, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_176
.LBB0_270:
	v_add_nc_u32_e32 v25, v67, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s16, v25
	v_add_co_ci_u32_e32 v26, vcc_lo, s17, v26, vcc_lo
	global_store_d16_hi_b16 v[25:26], v9, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_177
.LBB0_271:
	v_add_nc_u32_e32 v25, v67, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshlrev_b64 v[25:26], 1, v[25:26]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s16, v25
	v_add_co_ci_u32_e32 v26, vcc_lo, s17, v26, vcc_lo
	global_store_d16_hi_b16 v[25:26], v10, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s5, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_178
.LBB0_272:
	v_add_nc_u32_e32 v9, v67, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	v_add_co_ci_u32_e32 v10, vcc_lo, s17, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v11, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s6, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_179
.LBB0_273:
	v_add_nc_u32_e32 v9, v67, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	v_add_co_ci_u32_e32 v10, vcc_lo, s17, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v12, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_180
.LBB0_274:
	v_add_nc_u32_e32 v9, v67, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	v_add_co_ci_u32_e32 v10, vcc_lo, s17, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v13, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_181
.LBB0_275:
	v_add_nc_u32_e32 v9, v67, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	v_add_co_ci_u32_e32 v10, vcc_lo, s17, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v14, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB0_182
.LBB0_276:
	v_add_nc_u32_e32 v9, v67, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	v_add_co_ci_u32_e32 v10, vcc_lo, s17, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v15, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_183
.LBB0_277:
	v_add_nc_u32_e32 v9, v67, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	v_add_co_ci_u32_e32 v10, vcc_lo, s17, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v16, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_184
.LBB0_278:
	v_add_nc_u32_e32 v9, v57, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v10, 31, v9
	v_lshlrev_b64 v[9:10], 1, v[9:10]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	v_add_co_ci_u32_e32 v10, vcc_lo, s17, v10, vcc_lo
	global_store_d16_hi_b16 v[9:10], v1, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s4, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_185
.LBB0_279:
	v_add_nc_u32_e32 v0, v57, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s16, v0
	v_add_co_ci_u32_e32 v1, vcc_lo, s17, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v2, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s5, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_186
.LBB0_280:
	v_add_nc_u32_e32 v0, v57, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s16, v0
	v_add_co_ci_u32_e32 v1, vcc_lo, s17, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v3, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s6, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_187
.LBB0_281:
	v_add_nc_u32_e32 v0, v57, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s16, v0
	v_add_co_ci_u32_e32 v1, vcc_lo, s17, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v4, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_188
.LBB0_282:
	v_add_nc_u32_e32 v0, v57, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s16, v0
	v_add_co_ci_u32_e32 v1, vcc_lo, s17, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v5, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_189
.LBB0_283:
	v_add_nc_u32_e32 v0, v57, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s16, v0
	v_add_co_ci_u32_e32 v1, vcc_lo, s17, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v6, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_190
.LBB0_284:
	v_add_nc_u32_e32 v0, v57, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s16, v0
	v_add_co_ci_u32_e32 v1, vcc_lo, s17, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v7, off
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s0, s10, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_191
.LBB0_285:
	v_add_nc_u32_e32 v0, v57, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s16, v0
	v_add_co_ci_u32_e32 v1, vcc_lo, s17, v1, vcc_lo
	global_store_d16_hi_b16 v[0:1], v8, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii
		.amdhsa_group_segment_fixed_size 30720
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
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 22
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
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii,"axG",@progbits,_ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii,comdat
.Lfunc_end0:
	.size	_ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii, .Lfunc_end0-_ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii
                                        ; -- End function
	.set _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii.num_vgpr, 151
	.set _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii.num_agpr, 0
	.set _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii.numbered_sgpr, 22
	.set _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii.private_seg_size, 0
	.set _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii.uses_vcc, 1
	.set _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii.uses_flat_scratch, 0
	.set _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii.has_dyn_sized_stack, 0
	.set _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii.has_recursion, 0
	.set _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14132
; TotalNumSgprs: 24
; NumVgprs: 151
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 30720 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 169
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_90c0c619dc835f4b,@object ; @__hip_cuid_90c0c619dc835f4b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_90c0c619dc835f4b
__hip_cuid_90c0c619dc835f4b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_90c0c619dc835f4b, 1

	.ident	"clang version 20.0.0.rocm"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_90c0c619dc835f4b
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
    .group_segment_fixed_size: 30720
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         _ZN14rocm_wmma_gemm18block_k2_ring_gemm3runEP6__halfPKS1_S4_iii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     128
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 0
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
