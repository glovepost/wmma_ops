// k_i4r_rows<640> — i4r GEMV for expert down_proj (K=640)
// Disassembled from halogen-tools (ghcr.io/peonist-ai/halogen-flash-server:0.18.1),
// gfx1151 code object, llvm-objdump 23 --mcpu=gfx1151. Peonist's code, kept for internal
// reference only (their EULA permits RE; do not redistribute as our own work).
// Annotations: see docs/quantized_weight_decode_gfx1151.md.

0000000000e10100 <_ZN7halogen12_GLOBAL__N_110k_i4r_rowsILi640EEEvPKhliPKDF16_iPKiPf>:
s_load_b32 s10, s[0:1], 0x10
s_ashr_i32 s3, s10, 31
s_lshr_b32 s3, s3, 26
s_add_i32 s3, s10, s3
s_ashr_i32 s3, s3, 6
s_cvt_f32_u32 s4, s3
s_sub_i32 s5, 0, s3
v_rcp_iflag_f32_e32 v1, s4
v_readfirstlane_b32 s4, v1
s_mul_f32 s4, s4, 0x4f7ffffe
s_cvt_u32_f32 s4, s4
s_mul_i32 s5, s5, s4
s_mul_hi_u32 s5, s4, s5
s_add_i32 s4, s4, s5
s_mul_hi_u32 s4, s2, s4
s_mul_i32 s5, s4, s3
s_add_i32 s6, s4, 1
s_sub_i32 s5, s2, s5
s_sub_i32 s7, s5, s3
s_cmp_ge_u32 s5, s3
s_cselect_b32 s4, s6, s4
s_cselect_b32 s5, s7, s5
s_add_i32 s6, s4, 1
s_cmp_ge_u32 s5, s3
s_cselect_b32 s8, s6, s4
s_mov_b32 s4, exec_lo
v_cmpx_gt_u32_e32 0x50, v0
s_cbranch_execz 51
s_load_b32 s5, s[0:1], 0x20
s_abs_i32 s11, s8
v_lshlrev_b32_e32 v5, 4, v0
s_abs_i32 s9, s5
s_xor_b32 s5, s8, s5
s_cvt_f32_u32 s6, s9
s_sub_i32 s7, 0, s9
s_ashr_i32 s5, s5, 31
v_rcp_iflag_f32_e32 v1, s6
v_readfirstlane_b32 s6, v1
s_mul_f32 s6, s6, 0x4f7ffffe
s_cvt_u32_f32 s6, s6
s_mul_i32 s7, s7, s6
s_mul_hi_u32 s7, s6, s7
s_add_i32 s12, s6, s7
s_load_b64 s[6:7], s[0:1], 0x18
s_mul_hi_u32 s12, s11, s12
s_mul_i32 s13, s12, s9
s_sub_i32 s11, s11, s13
s_add_i32 s13, s12, 1
s_sub_i32 s14, s11, s9
s_cmp_ge_u32 s11, s9
s_cselect_b32 s12, s13, s12
s_cselect_b32 s11, s14, s11
s_add_i32 s13, s12, 1
s_cmp_ge_u32 s11, s9
s_cselect_b32 s9, s13, s12
s_xor_b32 s9, s9, s5
s_sub_i32 s5, s9, s5
s_mul_i32 s9, s5, 0x500
s_mul_hi_i32 s5, s5, 0x500
s_add_u32 s6, s6, s9
s_addc_u32 s7, s7, s5
global_load_b128 v[1:4], v5, s[6:7]
ds_store_b128 v5, v[1:4]
s_or_b32 exec_lo, exec_lo, s4
s_load_b128 s[4:7], s[0:1], 0x28
s_ashr_i32 s9, s8, 31
s_mul_i32 s3, s8, s3
s_lshl_b64 s[12:13], s[8:9], 2
s_sub_i32 s9, s2, s3
s_barrier
buffer_gl0_inv
v_lshrrev_b32_e32 v1, 2, v0
v_bfe_u32 v2, v0, 2, 3
v_mov_b32_e32 v30, 0
v_dual_mov_b32 v29, 0 :: v_dual_and_b32 v28, 3, v0
v_and_b32_e32 v1, 56, v1
v_mov_b32_e32 v31, 0
v_dual_mov_b32 v33, 0 :: v_dual_lshlrev_b32 v32, 6, v28
v_lshlrev_b32_e32 v0, 4, v28
ds_load_b128 v[47:50], v32
ds_load_b128 v[51:54], v32 offset:16
s_add_u32 s2, s4, s12
s_addc_u32 s3, s5, s13
s_lshl_b32 s5, s9, 6
s_load_b32 s4, s[2:3], null
s_load_b128 s[0:3], s[0:1], null
v_or3_b32 v24, s5, v1, v2
ds_load_b128 v[55:58], v32 offset:256
ds_load_b128 v[59:62], v32 offset:272
ds_load_b128 v[20:23], v32 offset:288
ds_load_b128 v[12:15], v32 offset:304
ds_load_b128 v[63:66], v32 offset:512
ds_load_b128 v[67:70], v32 offset:528
ds_load_b128 v[71:74], v32 offset:768
ds_load_b128 v[16:19], v32 offset:32
ds_load_b128 v[8:11], v32 offset:48
v_ashrrev_i32_e32 v25, 31, v24
v_mad_i64_i32 v[26:27], null, s4, s10, v[24:25]
v_mad_u64_u32 v[1:2], null, 0x140, v26, s[0:1]
s_add_u32 s0, s0, s2
s_addc_u32 s1, s1, s3
v_mad_u64_u32 v[75:76], null, v26, 10, s[0:1]
s_mov_b32 s0, exec_lo
v_mad_u64_u32 v[2:3], null, 0x140, v27, v[2:3]
v_add_co_u32 v0, vcc_lo, v1, v0
v_mad_u64_u32 v[76:77], null, v27, 10, v[76:77]
v_add_co_ci_u32_e64 v1, null, 0, v2, vcc_lo
s_clause 0x4
global_load_b128 v[35:38], v[0:1], off
global_load_b128 v[39:42], v[0:1], off offset:64
global_load_b128 v[43:46], v[0:1], off offset:128
global_load_b128 v[4:7], v[0:1], off offset:192
global_load_b128 v[0:3], v[0:1], off offset:256
v_dual_mov_b32 v34, 0 :: v_dual_and_b32 v77, 0xf0f0f0f, v35
v_lshrrev_b32_e32 v26, 4, v35
v_and_b32_e32 v78, 0xf0f0f0f, v36
v_lshrrev_b32_e32 v79, 4, v36
v_and_b32_e32 v80, 0xf0f0f0f, v37
v_lshrrev_b32_e32 v81, 4, v37
v_and_b32_e32 v36, 0xf0f0f0f, v39
v_lshrrev_b32_e32 v37, 4, v39
v_and_b32_e32 v39, 0xf0f0f0f, v26
v_and_b32_e32 v88, 0xf0f0f0f, v42
v_lshrrev_b32_e32 v89, 4, v42
v_and_b32_e32 v92, 0xf0f0f0f, v45
v_lshrrev_b32_e32 v93, 4, v45
v_perm_b32 v26, v39, v77, 0xc040c00
v_lshrrev_b32_e32 v87, 4, v41
v_and_b32_e32 v82, 0xf0f0f0f, v38
v_lshrrev_b32_e32 v83, 4, v38
v_and_b32_e32 v84, 0xf0f0f0f, v40
v_or_b32_e32 v42, 0x64006400, v26
s_clause 0x1
global_load_d16_b16 v35, v[75:76], off offset:8
global_load_b64 v[26:27], v[75:76], off
v_lshrrev_b32_e32 v85, 4, v40
v_and_b32_e32 v38, 0xf0f0f0f, v43
v_lshrrev_b32_e32 v40, 4, v43
v_pk_add_f16 v42, 0xe408, v42 op_sel_hi:[0,1]
v_and_b32_e32 v90, 0xf0f0f0f, v44
v_lshrrev_b32_e32 v91, 4, v44
v_and_b32_e32 v44, 0xf0f0f0f, v4
v_lshrrev_b32_e32 v4, 4, v4
v_dual_dot2acc_f32_f16 v30, v42, v47 :: v_dual_and_b32 v37, 0xf0f0f0f, v37
v_perm_b32 v43, v39, v77, 0xc060c02
v_and_b32_e32 v40, 0xf0f0f0f, v40
v_and_b32_e32 v4, 0xf0f0f0f, v4
v_and_b32_e32 v94, 0xf0f0f0f, v46
v_perm_b32 v45, v37, v36, 0xc070c03
v_or_b32_e32 v43, 0x64006400, v43
v_lshrrev_b32_e32 v95, 4, v46
v_perm_b32 v46, v40, v38, 0xc050c01
v_perm_b32 v47, v40, v38, 0xc070c03
v_or_b32_e32 v45, 0x64006400, v45
v_pk_add_f16 v42, 0xe408, v43 op_sel_hi:[0,1]
v_and_b32_e32 v75, 0xf0f0f0f, v87
v_pk_add_f16 v45, 0xe408, v45 op_sel_hi:[0,1]
v_and_b32_e32 v86, 0xf0f0f0f, v41
v_perm_b32 v41, v39, v77, 0xc050c01
v_perm_b32 v39, v39, v77, 0xc070c03
v_or_b32_e32 v41, 0x64006400, v41
v_or_b32_e32 v39, 0x64006400, v39
v_pk_add_f16 v41, 0xe408, v41 op_sel_hi:[0,1]
v_pk_add_f16 v39, 0xe408, v39 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v30, v41, v48
v_perm_b32 v41, v37, v36, 0xc040c00
v_perm_b32 v43, v37, v36, 0xc050c01
v_perm_b32 v48, v4, v44, 0xc050c01
v_dot2acc_f32_f16 v30, v42, v49
v_perm_b32 v42, v37, v36, 0xc060c02
v_perm_b32 v36, v40, v38, 0xc040c00
v_or_b32_e32 v37, 0x64006400, v41
v_perm_b32 v41, v40, v38, 0xc060c02
v_perm_b32 v40, v4, v44, 0xc040c00
v_perm_b32 v49, v4, v44, 0xc060c02
v_or_b32_e32 v36, 0x64006400, v36
v_pk_add_f16 v37, 0xe408, v37 op_sel_hi:[0,1]
v_perm_b32 v4, v4, v44, 0xc070c03
v_or_b32_e32 v40, 0x64006400, v40
v_or_b32_e32 v43, 0x64006400, v43
v_pk_add_f16 v36, 0xe408, v36 op_sel_hi:[0,1]
v_or_b32_e32 v44, 0x64006400, v46
v_or_b32_e32 v46, 0x64006400, v48
v_pk_add_f16 v40, 0xe408, v40 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v31, v37, v55
v_dot2acc_f32_f16 v33, v36, v63
v_or_b32_e32 v42, 0x64006400, v42
v_or_b32_e32 v41, 0x64006400, v41
v_pk_add_f16 v43, 0xe408, v43 op_sel_hi:[0,1]
v_pk_add_f16 v44, 0xe408, v44 op_sel_hi:[0,1]
v_pk_add_f16 v46, 0xe408, v46 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v34, v40, v71
v_pk_add_f16 v48, 0xe408, v42 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v31, v43, v56
v_dot2acc_f32_f16 v33, v44, v64
v_or_b32_e32 v44, 0x64006400, v49
v_dual_dot2acc_f32_f16 v34, v46, v72 :: v_dual_and_b32 v49, 0xf0f0f0f, v79
v_dot2acc_f32_f16 v30, v39, v50
v_pk_add_f16 v50, 0xe408, v41 op_sel_hi:[0,1]
v_or_b32_e32 v46, 0x64006400, v47
v_dual_dot2acc_f32_f16 v31, v48, v57 :: v_dual_and_b32 v48, 0xf0f0f0f, v0
v_lshrrev_b32_e32 v0, 4, v0
v_dot2acc_f32_f16 v33, v50, v65
v_pk_add_f16 v46, 0xe408, v46 op_sel_hi:[0,1]
ds_load_b128 v[36:39], v32 offset:784
ds_load_b128 v[40:43], v32 offset:1024
v_pk_add_f16 v44, 0xe408, v44 op_sel_hi:[0,1]
v_or_b32_e32 v4, 0x64006400, v4
v_dual_dot2acc_f32_f16 v33, v46, v66 :: v_dual_and_b32 v0, 0xf0f0f0f, v0
v_perm_b32 v71, v49, v78, 0xc040c00
v_dot2acc_f32_f16 v34, v44, v73
v_pk_add_f16 v4, 0xe408, v4 op_sel_hi:[0,1]
v_perm_b32 v72, v49, v78, 0xc060c02
v_perm_b32 v50, v0, v48, 0xc040c00
v_perm_b32 v63, v0, v48, 0xc050c01
v_dot2acc_f32_f16 v31, v45, v58
v_dual_dot2acc_f32_f16 v34, v4, v74 :: v_dual_and_b32 v79, 0xf0f0f0f, v93
v_or_b32_e32 v50, 0x64006400, v50
v_perm_b32 v4, v49, v78, 0xc050c01
v_or_b32_e32 v73, 0x64006400, v63
v_perm_b32 v49, v49, v78, 0xc070c03
ds_load_b128 v[63:66], v32 offset:1040
v_pk_add_f16 v50, 0xe408, v50 op_sel_hi:[0,1]
v_or_b32_e32 v4, 0x64006400, v4
v_and_b32_e32 v74, 0xf0f0f0f, v83
ds_load_b128 v[44:47], v32 offset:544
ds_load_b128 v[55:58], v32 offset:560
v_dual_dot2acc_f32_f16 v29, v50, v40 :: v_dual_and_b32 v78, 0xf0f0f0f, v91
v_or_b32_e32 v50, 0x64006400, v71
v_perm_b32 v40, v0, v48, 0xc060c02
v_pk_add_f16 v71, 0xe408, v73 op_sel_hi:[0,1]
v_perm_b32 v0, v0, v48, 0xc070c03
v_and_b32_e32 v48, 0xf0f0f0f, v5
v_pk_add_f16 v50, 0xe408, v50 op_sel_hi:[0,1]
v_pk_add_f16 v4, 0xe408, v4 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v29, v71, v41
v_lshrrev_b32_e32 v71, 4, v5
v_or_b32_e32 v5, 0x64006400, v40
v_dot2acc_f32_f16 v30, v50, v51
v_or_b32_e32 v40, 0x64006400, v72
v_and_b32_e32 v50, 0xf0f0f0f, v6
v_or_b32_e32 v0, 0x64006400, v0
v_pk_add_f16 v5, 0xe408, v5 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v30, v4, v52
v_lshrrev_b32_e32 v51, 4, v6
v_or_b32_e32 v4, 0x64006400, v49
v_pk_add_f16 v6, 0xe408, v40 op_sel_hi:[0,1]
v_pk_add_f16 v0, 0xe408, v0 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v29, v5, v42
v_and_b32_e32 v73, 0xf0f0f0f, v81
v_pk_add_f16 v72, 0xe408, v4 op_sel_hi:[0,1]
v_dual_dot2acc_f32_f16 v30, v6, v53 :: v_dual_and_b32 v81, 0xf0f0f0f, v95
v_and_b32_e32 v71, 0xf0f0f0f, v71
v_dot2acc_f32_f16 v29, v0, v43
v_and_b32_e32 v53, 0xf0f0f0f, v2
v_lshrrev_b32_e32 v2, 4, v2
v_dual_dot2acc_f32_f16 v30, v72, v54 :: v_dual_and_b32 v51, 0xf0f0f0f, v51
v_and_b32_e32 v72, 0xf0f0f0f, v85
v_lshrrev_b32_e32 v52, 4, v7
v_and_b32_e32 v2, 0xf0f0f0f, v2
v_and_b32_e32 v49, 0xf0f0f0f, v7
ds_load_b128 v[4:7], v32 offset:800
ds_load_b128 v[40:43], v32 offset:816
v_perm_b32 v77, v72, v84, 0xc040c00
v_perm_b32 v83, v72, v84, 0xc050c01
v_perm_b32 v85, v72, v84, 0xc060c02
v_perm_b32 v72, v72, v84, 0xc070c03
v_and_b32_e32 v52, 0xf0f0f0f, v52
v_or_b32_e32 v77, 0x64006400, v77
v_or_b32_e32 v83, 0x64006400, v83
v_or_b32_e32 v84, 0x64006400, v85
v_or_b32_e32 v72, 0x64006400, v72
v_perm_b32 v85, v75, v86, 0xc060c02
v_pk_add_f16 v77, 0xe408, v77 op_sel_hi:[0,1]
v_pk_add_f16 v83, 0xe408, v83 op_sel_hi:[0,1]
v_pk_add_f16 v84, 0xe408, v84 op_sel_hi:[0,1]
v_pk_add_f16 v72, 0xe408, v72 op_sel_hi:[0,1]
v_perm_b32 v93, v2, v53, 0xc060c02
v_dot2acc_f32_f16 v31, v77, v59
v_perm_b32 v59, v73, v80, 0xc040c00
v_perm_b32 v77, v73, v80, 0xc050c01
v_dot2acc_f32_f16 v31, v83, v60
v_perm_b32 v60, v73, v80, 0xc060c02
v_perm_b32 v73, v73, v80, 0xc070c03
v_perm_b32 v80, v74, v82, 0xc040c00
v_perm_b32 v83, v74, v82, 0xc060c02
v_dot2acc_f32_f16 v31, v84, v61
v_perm_b32 v61, v74, v82, 0xc050c01
v_perm_b32 v74, v74, v82, 0xc070c03
v_perm_b32 v82, v78, v90, 0xc040c00
v_perm_b32 v84, v78, v90, 0xc050c01
v_dot2acc_f32_f16 v31, v72, v62
v_perm_b32 v62, v75, v86, 0xc040c00
v_perm_b32 v72, v75, v86, 0xc050c01
v_or_b32_e32 v82, 0x64006400, v82
v_perm_b32 v75, v75, v86, 0xc070c03
v_perm_b32 v86, v78, v90, 0xc060c02
v_or_b32_e32 v84, 0x64006400, v84
v_perm_b32 v78, v78, v90, 0xc070c03
v_pk_add_f16 v82, 0xe408, v82 op_sel_hi:[0,1]
v_perm_b32 v90, v71, v48, 0xc060c02
v_or_b32_e32 v86, 0x64006400, v86
v_pk_add_f16 v84, 0xe408, v84 op_sel_hi:[0,1]
v_or_b32_e32 v78, 0x64006400, v78
v_dot2acc_f32_f16 v33, v82, v67
v_pk_add_f16 v86, 0xe408, v86 op_sel_hi:[0,1]
v_pk_add_f16 v78, 0xe408, v78 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v33, v84, v68
v_perm_b32 v84, v79, v92, 0xc050c01
v_dot2acc_f32_f16 v33, v86, v69
v_perm_b32 v86, v71, v48, 0xc040c00
v_and_b32_e32 v0, 0xf0f0f0f, v1
v_lshrrev_b32_e32 v1, 4, v1
v_perm_b32 v91, v51, v50, 0xc060c02
v_dot2acc_f32_f16 v33, v78, v70
v_perm_b32 v78, v71, v48, 0xc050c01
v_or_b32_e32 v86, 0x64006400, v86
v_perm_b32 v48, v71, v48, 0xc070c03
v_or_b32_e32 v71, 0x64006400, v90
v_perm_b32 v90, v51, v50, 0xc050c01
v_or_b32_e32 v78, 0x64006400, v78
v_pk_add_f16 v86, 0xe408, v86 op_sel_hi:[0,1]
v_and_b32_e32 v1, 0xf0f0f0f, v1
v_perm_b32 v69, v79, v92, 0xc060c02
v_perm_b32 v70, v81, v94, 0xc040c00
v_pk_add_f16 v78, 0xe408, v78 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v34, v86, v36
v_pk_add_f16 v36, 0xe408, v71 op_sel_hi:[0,1]
v_perm_b32 v86, v51, v50, 0xc040c00
v_perm_b32 v50, v51, v50, 0xc070c03
v_perm_b32 v51, v52, v49, 0xc050c01
v_dot2acc_f32_f16 v34, v78, v37
v_perm_b32 v78, v2, v53, 0xc040c00
v_or_b32_e32 v37, 0x64006400, v48
v_perm_b32 v48, v52, v49, 0xc040c00
v_perm_b32 v71, v52, v49, 0xc060c02
v_perm_b32 v49, v52, v49, 0xc070c03
v_or_b32_e32 v78, 0x64006400, v78
v_perm_b32 v52, v1, v0, 0xc070c03
v_or_b32_e32 v69, 0x64006400, v69
v_or_b32_e32 v70, 0x64006400, v70
v_or_b32_e32 v50, 0x64006400, v50
v_pk_add_f16 v78, 0xe408, v78 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v34, v36, v38
v_pk_add_f16 v36, 0xe408, v37 op_sel_hi:[0,1]
v_perm_b32 v37, v1, v0, 0xc040c00
v_perm_b32 v38, v1, v0, 0xc060c02
v_or_b32_e32 v52, 0x64006400, v52
v_pk_add_f16 v69, 0xe408, v69 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v34, v36, v39
v_or_b32_e32 v37, 0x64006400, v37
v_perm_b32 v36, v1, v0, 0xc050c01
v_or_b32_e32 v1, 0x64006400, v59
v_pk_add_f16 v52, 0xe408, v52 op_sel_hi:[0,1]
v_or_b32_e32 v48, 0x64006400, v48
v_pk_add_f16 v0, 0xe408, v37 op_sel_hi:[0,1]
v_or_b32_e32 v37, 0x64006400, v38
v_pk_add_f16 v1, 0xe408, v1 op_sel_hi:[0,1]
v_pk_add_f16 v70, 0xe408, v70 op_sel_hi:[0,1]
v_pk_add_f16 v50, 0xe408, v50 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v29, v0, v63
v_or_b32_e32 v0, 0x64006400, v36
v_or_b32_e32 v36, 0x64006400, v62
v_pk_add_f16 v37, 0xe408, v37 op_sel_hi:[0,1]
v_or_b32_e32 v62, 0x64006400, v75
v_or_b32_e32 v59, 0x64006400, v74
v_pk_add_f16 v0, 0xe408, v0 op_sel_hi:[0,1]
v_and_b32_e32 v54, 0xf0f0f0f, v3
v_lshrrev_b32_e32 v3, 4, v3
v_pk_add_f16 v62, 0xe408, v62 op_sel_hi:[0,1]
v_or_b32_e32 v51, 0x64006400, v51
v_dot2acc_f32_f16 v29, v0, v64
v_pk_add_f16 v0, 0xe408, v36 op_sel_hi:[0,1]
v_and_b32_e32 v76, 0xf0f0f0f, v89
v_and_b32_e32 v3, 0xf0f0f0f, v3
v_perm_b32 v89, v81, v94, 0xc060c02
v_dot2acc_f32_f16 v29, v37, v65
v_dot2acc_f32_f16 v31, v0, v20
v_perm_b32 v87, v76, v88, 0xc040c00
v_perm_b32 v67, v76, v88, 0xc050c01
v_perm_b32 v82, v76, v88, 0xc060c02
v_perm_b32 v68, v76, v88, 0xc070c03
v_perm_b32 v76, v79, v92, 0xc040c00
v_perm_b32 v79, v79, v92, 0xc070c03
v_perm_b32 v92, v2, v53, 0xc050c01
v_perm_b32 v53, v2, v53, 0xc070c03
v_or_b32_e32 v2, 0x64006400, v77
v_or_b32_e32 v38, 0x64006400, v76
v_dual_dot2acc_f32_f16 v30, v1, v16 :: v_dual_dot2acc_f32_f16 v29, v52, v66
v_perm_b32 v88, v81, v94, 0xc050c01
v_pk_add_f16 v2, 0xe408, v2 op_sel_hi:[0,1]
v_pk_add_f16 v36, 0xe408, v38 op_sel_hi:[0,1]
v_perm_b32 v81, v81, v94, 0xc070c03
v_perm_b32 v94, v3, v54, 0xc040c00
v_perm_b32 v95, v3, v54, 0xc050c01
v_perm_b32 v63, v3, v54, 0xc060c02
v_perm_b32 v16, v3, v54, 0xc070c03
v_dual_dot2acc_f32_f16 v33, v36, v44 :: v_dual_dot2acc_f32_f16 v30, v2, v17
ds_load_b128 v[0:3], v32 offset:1056
ds_load_b128 v[36:39], v32 offset:1072
v_or_b32_e32 v17, 0x64006400, v60
v_or_b32_e32 v60, 0x64006400, v72
v_or_b32_e32 v72, 0x64006400, v84
v_or_b32_e32 v44, 0x64006400, v61
v_or_b32_e32 v61, 0x64006400, v85
v_or_b32_e32 v20, 0x64006400, v73
v_pk_add_f16 v60, 0xe408, v60 op_sel_hi:[0,1]
v_pk_add_f16 v72, 0xe408, v72 op_sel_hi:[0,1]
v_or_b32_e32 v76, 0x64006400, v81
v_or_b32_e32 v77, 0x64006400, v86
v_or_b32_e32 v81, 0x64006400, v92
v_pk_add_f16 v17, 0xe408, v17 op_sel_hi:[0,1]
v_pk_add_f16 v61, 0xe408, v61 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v31, v60, v21
v_dot2acc_f32_f16 v33, v72, v45
v_or_b32_e32 v65, 0x64006400, v67
v_or_b32_e32 v67, 0x64006400, v82
v_or_b32_e32 v73, 0x64006400, v79
v_or_b32_e32 v79, 0x64006400, v90
v_or_b32_e32 v82, 0x64006400, v93
v_pk_add_f16 v20, 0xe408, v20 op_sel_hi:[0,1]
v_pk_add_f16 v77, 0xe408, v77 op_sel_hi:[0,1]
v_pk_add_f16 v81, 0xe408, v81 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v31, v61, v22
v_dot2acc_f32_f16 v29, v78, v0
v_dot2acc_f32_f16 v33, v69, v46
v_dot2acc_f32_f16 v30, v17, v18
v_or_b32_e32 v32, 0x64006400, v80
v_or_b32_e32 v64, 0x64006400, v87
v_or_b32_e32 v80, 0x64006400, v91
v_pk_add_f16 v79, 0xe408, v79 op_sel_hi:[0,1]
v_pk_add_f16 v82, 0xe408, v82 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v31, v62, v23
v_dual_dot2acc_f32_f16 v30, v20, v19 :: v_dual_dot2acc_f32_f16 v29, v81, v1
v_dot2acc_f32_f16 v34, v77, v4
v_pk_add_f16 v32, 0xe408, v32 op_sel_hi:[0,1]
v_pk_add_f16 v64, 0xe408, v64 op_sel_hi:[0,1]
v_pk_add_f16 v73, 0xe408, v73 op_sel_hi:[0,1]
v_pk_add_f16 v80, 0xe408, v80 op_sel_hi:[0,1]
v_dual_dot2acc_f32_f16 v29, v82, v2 :: v_dual_dot2acc_f32_f16 v34, v79, v5
v_or_b32_e32 v54, 0x64006400, v83
v_or_b32_e32 v53, 0x64006400, v53
v_pk_add_f16 v44, 0xe408, v44 op_sel_hi:[0,1]
v_pk_add_f16 v65, 0xe408, v65 op_sel_hi:[0,1]
v_dual_dot2acc_f32_f16 v33, v73, v47 :: v_dual_dot2acc_f32_f16 v30, v32, v8
v_dot2acc_f32_f16 v31, v64, v12
v_dot2acc_f32_f16 v34, v80, v6
v_or_b32_e32 v74, 0x64006400, v88
v_or_b32_e32 v83, 0x64006400, v94
v_pk_add_f16 v54, 0xe408, v54 op_sel_hi:[0,1]
v_pk_add_f16 v48, 0xe408, v48 op_sel_hi:[0,1]
v_pk_add_f16 v53, 0xe408, v53 op_sel_hi:[0,1]
v_dual_dot2acc_f32_f16 v33, v70, v55 :: v_dual_dot2acc_f32_f16 v30, v44, v9
v_dual_dot2acc_f32_f16 v31, v65, v13 :: v_dual_dot2acc_f32_f16 v34, v50, v7
v_or_b32_e32 v68, 0x64006400, v68
v_or_b32_e32 v75, 0x64006400, v89
v_or_b32_e32 v71, 0x64006400, v71
v_or_b32_e32 v84, 0x64006400, v95
v_pk_add_f16 v59, 0xe408, v59 op_sel_hi:[0,1]
v_pk_add_f16 v67, 0xe408, v67 op_sel_hi:[0,1]
v_pk_add_f16 v74, 0xe408, v74 op_sel_hi:[0,1]
v_pk_add_f16 v51, 0xe408, v51 op_sel_hi:[0,1]
v_pk_add_f16 v83, 0xe408, v83 op_sel_hi:[0,1]
v_dual_dot2acc_f32_f16 v29, v53, v3 :: v_dual_dot2acc_f32_f16 v30, v54, v10
v_dual_dot2acc_f32_f16 v34, v48, v40 :: v_dual_dot2acc_f32_f16 v31, v67, v14
v_or_b32_e32 v49, 0x64006400, v49
v_pk_add_f16 v68, 0xe408, v68 op_sel_hi:[0,1]
v_pk_add_f16 v75, 0xe408, v75 op_sel_hi:[0,1]
v_pk_add_f16 v71, 0xe408, v71 op_sel_hi:[0,1]
v_pk_add_f16 v84, 0xe408, v84 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v29, v83, v36
v_dual_dot2acc_f32_f16 v33, v74, v56 :: v_dual_dot2acc_f32_f16 v30, v59, v11
v_dual_dot2acc_f32_f16 v34, v51, v41 :: v_dual_dot2acc_f32_f16 v31, v68, v15
v_or_b32_e32 v63, 0x64006400, v63
v_pk_add_f16 v76, 0xe408, v76 op_sel_hi:[0,1]
v_pk_add_f16 v49, 0xe408, v49 op_sel_hi:[0,1]
v_fma_mix_f32 v0, v26, v30, 0 op_sel_hi:[1,0,0]
v_dual_dot2acc_f32_f16 v29, v84, v37 :: v_dual_dot2acc_f32_f16 v34, v71, v42
v_dot2acc_f32_f16 v33, v75, v57
v_or_b32_e32 v16, 0x64006400, v16
v_pk_add_f16 v1, 0xe408, v63 op_sel_hi:[0,1]
v_fma_mix_f32 v0, v26, v31, v0 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_dual_dot2acc_f32_f16 v34, v49, v43 :: v_dual_dot2acc_f32_f16 v33, v76, v58
v_pk_add_f16 v2, 0xe408, v16 op_sel_hi:[0,1]
v_dot2acc_f32_f16 v29, v1, v38
v_fma_mix_f32 v0, v27, v33, v0 op_sel_hi:[1,0,0]
v_dot2acc_f32_f16 v29, v2, v39
v_fma_mix_f32 v0, v27, v34, v0 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v0, v35, v29, v0 op_sel_hi:[1,0,0]
v_mov_b32_dpp v1, v0 row_xmask:1 row_mask:0xf bank_mask:0xf
v_add_f32_e32 v0, v0, v1
v_mov_b32_dpp v1, v0 row_xmask:2 row_mask:0xf bank_mask:0xf
v_cmpx_eq_u32_e32 0, v28
s_cbranch_execz 15
s_mul_hi_i32 s1, s8, s10
s_mul_i32 s0, s8, s10
v_lshlrev_b64 v[2:3], 2, v[24:25]
s_lshl_b64 s[0:1], s[0:1], 2
v_add_f32_e32 v4, v0, v1
s_add_u32 s0, s6, s0
s_addc_u32 s1, s7, s1
v_add_co_u32 v0, vcc_lo, s0, v2
v_add_co_ci_u32_e64 v1, null, s1, v3, vcc_lo
global_store_b32 v[0:1], v4, off
s_endpgm
...
