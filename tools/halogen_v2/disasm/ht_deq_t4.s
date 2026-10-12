// k_ht_deq_t<4> — ht4 dequant (trellis, mul1 codebook)
// Disassembled from halogen-tools (ghcr.io/peonist-ai/halogen-flash-server:0.18.1),
// gfx1151 code object, llvm-objdump 23 --mcpu=gfx1151. Peonist's code, kept for internal
// reference only (their EULA permits RE; do not redistribute as our own work).
// Annotations: see docs/quantized_weight_decode_gfx1151.md.


b00.co:	file format elf64-amdgpu
	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx1151"

Disassembly of section .text:

0000000000554d00 <_ZN7halogen12_GLOBAL__N_110k_ht_deq_tILi4EEEvPKjiiPt>:
	s_load_b64 s[4:5], s[0:1], 0x8                             // 000000554D00: F4040100 F8000008
	v_lshrrev_b32_e32 v1, 5, v0                                // 000000554D08: 32020085
	s_mov_b32 s3, 0                                            // 000000554D0C: BE830080
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000554D10: BF870499
	s_lshl_b64 s[2:3], s[2:3], 3                               // 000000554D14: 84828302
	v_mov_b32_e32 v2, s3                                       // 000000554D18: 7E040203
	s_delay_alu instid0(VALU_DEP_2)                            // 000000554D1C: BF870002
	v_or_b32_e32 v1, s2, v1                                    // 000000554D20: 38020202
	s_waitcnt lgkmcnt(0)                                       // 000000554D24: BF89FC07
	s_ashr_i32 s2, s4, 31                                      // 000000554D28: 86029F04
	s_ashr_i32 s3, s5, 31                                      // 000000554D2C: 86039F05
	s_lshr_b32 s2, s2, 28                                      // 000000554D30: 85029C02
	s_lshr_b32 s3, s3, 28                                      // 000000554D34: 85039C03
	s_add_i32 s2, s4, s2                                       // 000000554D38: 81020204
	s_add_i32 s3, s5, s3                                       // 000000554D3C: 81030305
	s_ashr_i32 s6, s2, 4                                       // 000000554D40: 86068402
	s_ashr_i32 s2, s3, 4                                       // 000000554D44: 86028403
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000554D48: BF8704A9
	s_mul_hi_i32 s3, s2, s6                                    // 000000554D4C: 97030602
	s_mul_i32 s2, s2, s6                                       // 000000554D50: 96020602
	v_cmp_gt_i64_e32 vcc_lo, s[2:3], v[1:2]                    // 000000554D54: 7CA80202
	s_and_saveexec_b32 s2, vcc_lo                              // 000000554D58: BE82206A
	s_cbranch_execz 496                                        // 000000554D5C: BFA501F0 <_ZN7halogen12_GLOBAL__N_110k_ht_deq_tILi4EEEvPKjiiPt+0x820>
	s_ashr_i32 s7, s6, 31                                      // 000000554D60: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000554D64: BF8700A9
	v_or_b32_e32 v3, s7, v2                                    // 000000554D68: 38060407
	s_lshl_b64 s[8:9], s[6:7], 3                               // 000000554D6C: 84888306
	v_cmp_ne_u32_e32 vcc_lo, 0, v3                             // 000000554D70: 7C9A0680
	s_and_saveexec_b32 s2, vcc_lo                              // 000000554D74: BE82206A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000554D78: BF870009
	s_xor_b32 s3, exec_lo, s2                                  // 000000554D7C: 8D03027E
	s_cbranch_execz 170                                        // 000000554D80: BFA500AA <_ZN7halogen12_GLOBAL__N_110k_ht_deq_tILi4EEEvPKjiiPt+0x32c>
	s_ashr_i32 s10, s9, 31                                     // 000000554D84: 860A9F09
	v_ashrrev_i32_e32 v8, 31, v2                               // 000000554D88: 3410049F
	s_add_u32 s12, s8, s10                                     // 000000554D8C: 800C0A08
	s_mov_b32 s11, s10                                         // 000000554D90: BE8B000A
	s_addc_u32 s13, s9, s10                                    // 000000554D94: 820D0A09
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000554D98: BF870499
	s_xor_b64 s[12:13], s[12:13], s[10:11]                     // 000000554D9C: 8D8C0A0C
	s_cvt_f32_u32 s2, s12                                      // 000000554DA0: BE82650C
	s_cvt_f32_u32 s5, s13                                      // 000000554DA4: BE85650D
	s_sub_u32 s7, 0, s12                                       // 000000554DA8: 80870C80
	s_subb_u32 s9, 0, s13                                      // 000000554DAC: 82890D80
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)// 000000554DB0: BF870599
	s_fmamk_f32 s2, s5, 0x4f800000, s2                         // 000000554DB4: A3020205 4F800000
	v_rcp_f32_e32 v3, s2                                       // 000000554DBC: 7E065402
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000554DC0: BF8700A5
	v_readfirstlane_b32 s2, v3                                 // 000000554DC4: 7E040503
	v_add_co_u32 v3, vcc_lo, v1, v8                            // 000000554DC8: D7006A03 02021101
	v_add_co_ci_u32_e64 v2, null, v2, v8, vcc_lo               // 000000554DD0: D5207C02 01AA1102
	s_mul_f32 s2, s2, 0x5f7ffffc                               // 000000554DD8: A202FF02 5F7FFFFC
	v_xor_b32_e32 v10, v3, v8                                  // 000000554DE0: 3A141103
	v_xor_b32_e32 v9, v2, v8                                   // 000000554DE4: 3A121102
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)// 000000554DE8: BF870599
	s_mul_f32 s5, s2, 0x2f800000                               // 000000554DEC: A205FF02 2F800000
	s_trunc_f32 s5, s5                                         // 000000554DF4: BE856205
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)// 000000554DF8: BF87052B
	s_fmamk_f32 s2, s5, 0xcf800000, s2                         // 000000554DFC: A3020205 CF800000
	s_cvt_u32_f32 s5, s5                                       // 000000554E04: BE856705
	s_cvt_u32_f32 s2, s2                                       // 000000554E08: BE826702
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_2)// 000000554E0C: BF87051A
	s_mul_i32 s11, s7, s5                                      // 000000554E10: 960B0507
	s_mul_hi_u32 s15, s7, s2                                   // 000000554E14: 968F0207
	s_mul_i32 s14, s9, s2                                      // 000000554E18: 960E0209
	s_add_i32 s11, s15, s11                                    // 000000554E1C: 810B0B0F
	s_mul_i32 s16, s7, s2                                      // 000000554E20: 96100207
	s_add_i32 s11, s11, s14                                    // 000000554E24: 810B0E0B
	s_mul_hi_u32 s15, s2, s16                                  // 000000554E28: 968F1002
	s_mul_i32 s18, s2, s11                                     // 000000554E2C: 96120B02
	s_mul_hi_u32 s17, s5, s16                                  // 000000554E30: 96911005
	s_mul_i32 s14, s5, s16                                     // 000000554E34: 960E1005
	s_mul_hi_u32 s16, s2, s11                                  // 000000554E38: 96900B02
	s_add_u32 s15, s15, s18                                    // 000000554E3C: 800F120F
	s_addc_u32 s16, 0, s16                                     // 000000554E40: 82101080
	s_mul_hi_u32 s19, s5, s11                                  // 000000554E44: 96930B05
	s_add_u32 s14, s15, s14                                    // 000000554E48: 800E0E0F
	s_mul_i32 s11, s5, s11                                     // 000000554E4C: 960B0B05
	s_addc_u32 s14, s16, s17                                   // 000000554E50: 820E1110
	s_addc_u32 s15, s19, 0                                     // 000000554E54: 820F8013
	s_add_u32 s11, s14, s11                                    // 000000554E58: 800B0B0E
	s_addc_u32 s14, 0, s15                                     // 000000554E5C: 820E0F80
	s_add_u32 s2, s2, s11                                      // 000000554E60: 80020B02
	s_addc_u32 s5, s5, s14                                     // 000000554E64: 82050E05
	s_mul_hi_u32 s11, s7, s2                                   // 000000554E68: 968B0207
	s_mul_i32 s14, s7, s2                                      // 000000554E6C: 960E0207
	s_mul_i32 s7, s7, s5                                       // 000000554E70: 96070507
	s_mul_i32 s9, s9, s2                                       // 000000554E74: 96090209
	s_add_i32 s7, s11, s7                                      // 000000554E78: 8107070B
	s_mul_i32 s11, s5, s14                                     // 000000554E7C: 960B0E05
	s_add_i32 s7, s7, s9                                       // 000000554E80: 81070907
	s_mul_hi_u32 s9, s2, s14                                   // 000000554E84: 96890E02
	s_mul_i32 s17, s2, s7                                      // 000000554E88: 96110702
	s_mul_hi_u32 s16, s2, s7                                   // 000000554E8C: 96900702
	s_add_u32 s9, s9, s17                                      // 000000554E90: 80091109
	s_mul_hi_u32 s15, s5, s14                                  // 000000554E94: 968F0E05
	s_addc_u32 s16, 0, s16                                     // 000000554E98: 82101080
	s_mul_hi_u32 s14, s5, s7                                   // 000000554E9C: 968E0705
	s_add_u32 s9, s9, s11                                      // 000000554EA0: 80090B09
	s_mul_i32 s7, s5, s7                                       // 000000554EA4: 96070705
	s_addc_u32 s9, s16, s15                                    // 000000554EA8: 82090F10
	s_addc_u32 s11, s14, 0                                     // 000000554EAC: 820B800E
	s_add_u32 s7, s9, s7                                       // 000000554EB0: 80070709
	s_addc_u32 s9, 0, s11                                      // 000000554EB4: 82090B80
	s_add_u32 s2, s2, s7                                       // 000000554EB8: 80020702
	s_addc_u32 s5, s5, s9                                      // 000000554EBC: 82050905
	v_mul_hi_u32 v11, v10, s2                                  // 000000554EC0: D72D000B 0200050A
	v_mad_u64_u32 v[2:3], null, v10, s5, 0                     // 000000554EC8: D6FE7C02 02000B0A
	v_mad_u64_u32 v[4:5], null, v9, s2, 0                      // 000000554ED0: D6FE7C04 02000509
	v_mad_u64_u32 v[6:7], null, v9, s5, 0                      // 000000554ED8: D6FE7C06 02000B09
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000554EE0: BF870093
	v_add_co_u32 v2, vcc_lo, v11, v2                           // 000000554EE4: D7006A02 0202050B
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo                // 000000554EEC: D5207C03 01AA0680
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000554EF4: BF870112
	v_add_co_u32 v2, vcc_lo, v2, v4                            // 000000554EF8: D7006A02 02020902
	v_add_co_ci_u32_e32 v2, vcc_lo, v3, v5, vcc_lo             // 000000554F00: 40040B03
	v_add_co_ci_u32_e32 v3, vcc_lo, 0, v7, vcc_lo              // 000000554F04: 40060E80
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000554F08: BF870092
	v_add_co_u32 v4, vcc_lo, v2, v6                            // 000000554F0C: D7006A04 02020D02
	v_add_co_ci_u32_e64 v5, null, 0, v3, vcc_lo                // 000000554F14: D5207C05 01AA0680
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000554F1C: BF8701A2
	v_mul_lo_u32 v6, s13, v4                                   // 000000554F20: D72C0006 0202080D
	v_mad_u64_u32 v[2:3], null, s12, v4, 0                     // 000000554F28: D6FE7C02 0202080C
	v_mul_lo_u32 v7, s12, v5                                   // 000000554F30: D72C0007 02020A0C
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000554F38: BF870112
	v_sub_co_u32 v2, vcc_lo, v10, v2                           // 000000554F3C: D7016A02 0202050A
	v_add3_u32 v3, v3, v7, v6                                  // 000000554F44: D6550003 041A0F03
	v_add_co_u32 v7, s2, v4, 2                                 // 000000554F4C: D7000207 02010504
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000554F54: BF870191
	v_add_co_ci_u32_e64 v10, null, 0, v5, s2                   // 000000554F58: D5207C0A 000A0A80
	v_sub_nc_u32_e32 v6, v9, v3                                // 000000554F60: 4C0C0709
	v_sub_co_u32 v11, s2, v2, s12                              // 000000554F64: D701020B 02001902
	v_sub_co_ci_u32_e64 v3, null, v9, v3, vcc_lo               // 000000554F6C: D5217C03 01AA0709
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000554F74: BF870193
	v_subrev_co_ci_u32_e64 v6, null, s13, v6, vcc_lo           // 000000554F78: D5227C06 01AA0C0D
	v_cmp_le_u32_e32 vcc_lo, s12, v11                          // 000000554F80: 7C96160C
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000554F84: BF870122
	v_subrev_co_ci_u32_e64 v6, null, 0, v6, s2                 // 000000554F88: D5227C06 000A0C80
	v_cndmask_b32_e64 v9, 0, -1, vcc_lo                        // 000000554F90: D5010009 01A98280
	v_cmp_le_u32_e32 vcc_lo, s13, v6                           // 000000554F98: 7C960C0D
	v_cndmask_b32_e64 v11, 0, -1, vcc_lo                       // 000000554F9C: D501000B 01A98280
	v_cmp_le_u32_e32 vcc_lo, s12, v2                           // 000000554FA4: 7C96040C
	v_cndmask_b32_e64 v2, 0, -1, vcc_lo                        // 000000554FA8: D5010002 01A98280
	v_cmp_le_u32_e32 vcc_lo, s13, v3                           // 000000554FB0: 7C96060D
	v_cndmask_b32_e64 v12, 0, -1, vcc_lo                       // 000000554FB4: D501000C 01A98280
	v_cmp_eq_u32_e32 vcc_lo, s13, v6                           // 000000554FBC: 7C940C0D
	v_cndmask_b32_e32 v6, v11, v9, vcc_lo                      // 000000554FC0: 020C130B
	v_add_co_u32 v9, vcc_lo, v4, 1                             // 000000554FC4: D7006A09 02010304
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)// 000000554FCC: BF8701D1
	v_add_co_ci_u32_e64 v11, null, 0, v5, vcc_lo               // 000000554FD0: D5207C0B 01AA0A80
	v_cmp_eq_u32_e32 vcc_lo, s13, v3                           // 000000554FD8: 7C94060D
	v_cndmask_b32_e32 v2, v12, v2, vcc_lo                      // 000000554FDC: 0204050C
	v_cmp_ne_u32_e32 vcc_lo, 0, v6                             // 000000554FE0: 7C9A0C80
	v_xor_b32_e32 v6, s10, v8                                  // 000000554FE4: 3A0C100A
	v_cmp_ne_u32_e64 s2, 0, v2                                 // 000000554FE8: D44D0002 02020480
	v_dual_cndmask_b32 v3, v11, v10 :: v_dual_cndmask_b32 v2, v9, v7// 000000554FF0: CA52150B 03020F09
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000554FF8: BF870111
	v_cndmask_b32_e64 v3, v5, v3, s2                           // 000000554FFC: D5010003 000A0705
	v_cndmask_b32_e64 v2, v4, v2, s2                           // 000000555004: D5010002 000A0504
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000055500C: BF870112
	v_xor_b32_e32 v4, v3, v6                                   // 000000555010: 3A080D03
	v_xor_b32_e32 v2, v2, v6                                   // 000000555014: 3A040D02
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000555018: BF870091
	v_sub_co_u32 v3, vcc_lo, v2, v6                            // 00000055501C: D7016A03 02020D02
	v_sub_co_ci_u32_e64 v2, null, v4, v6, vcc_lo               // 000000555024: D5217C02 01AA0D04
	s_and_not1_saveexec_b32 s2, s3                             // 00000055502C: BE823003
	s_cbranch_execz 31                                         // 000000555030: BFA5001F <_ZN7halogen12_GLOBAL__N_110k_ht_deq_tILi4EEEvPKjiiPt+0x3b0>
	v_cvt_f32_u32_e32 v2, s8                                   // 000000555034: 7E040C08
	s_sub_i32 s3, 0, s8                                        // 000000555038: 81830880
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)// 00000055503C: BF870291
	v_rcp_iflag_f32_e32 v2, v2                                 // 000000555040: 7E045702
	v_mul_f32_e32 v2, 0x4f7ffffe, v2                           // 000000555044: 100404FF 4F7FFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000055504C: BF870091
	v_cvt_u32_f32_e32 v2, v2                                   // 000000555050: 7E040F02
	v_mul_lo_u32 v3, s3, v2                                    // 000000555054: D72C0003 02020403
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000055505C: BF870091
	v_mul_hi_u32 v3, v2, v3                                    // 000000555060: D72D0003 02020702
	v_add_nc_u32_e32 v2, v2, v3                                // 000000555068: 4A040702
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000055506C: BF870091
	v_mul_hi_u32 v2, v1, v2                                    // 000000555070: D72D0002 02020501
	v_mul_lo_u32 v3, v2, s8                                    // 000000555078: D72C0003 02001102
	v_add_nc_u32_e32 v4, 1, v2                                 // 000000555080: 4A080481
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000555084: BF870092
	v_sub_nc_u32_e32 v3, v1, v3                                // 000000555088: 4C060701
	v_subrev_nc_u32_e32 v5, s8, v3                             // 00000055508C: 4E0A0608
	v_cmp_le_u32_e32 vcc_lo, s8, v3                            // 000000555090: 7C960608
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000555094: BF870092
	v_dual_cndmask_b32 v3, v3, v5 :: v_dual_cndmask_b32 v2, v2, v4// 000000555098: CA520B03 03020902
	v_cmp_le_u32_e32 vcc_lo, s8, v3                            // 0000005550A0: 7C960608
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000005550A4: BF870092
	v_add_nc_u32_e32 v4, 1, v2                                 // 0000005550A8: 4A080481
	v_cndmask_b32_e32 v3, v2, v4, vcc_lo                       // 0000005550AC: 02060902
	s_or_b32 exec_lo, exec_lo, s2                              // 0000005550B0: 8C7E027E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000005550B4: BF8700C1
	v_mul_lo_u32 v2, s8, v3                                    // 0000005550B8: D72C0002 02020608
	s_clause 0x1                                               // 0000005550C0: BF850001
	s_load_b64 s[2:3], s[0:1], null                            // 0000005550C4: F4040080 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 0000005550CC: F4040000 F8000010
	v_sub_nc_u32_e32 v6, v1, v2                                // 0000005550D4: 4C0C0501
	v_lshlrev_b32_e32 v2, 3, v0                                // 0000005550D8: 30040083
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000005550DC: BF870112
	v_ashrrev_i32_e32 v1, 31, v6                               // 0000005550E0: 34020C9F
	v_and_b32_e32 v8, 0xf8, v2                                 // 0000005550E4: 361004FF 000000F8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000005550EC: BF870112
	v_lshrrev_b32_e32 v1, 29, v1                               // 0000005550F0: 3202029D
	v_lshlrev_b32_e32 v11, 2, v8                               // 0000005550F4: 30161082
	v_lshrrev_b32_e32 v10, 1, v8                               // 0000005550F8: 32141081
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000005550FC: BF870193
	v_add_nc_u32_e32 v4, v6, v1                                // 000000555100: 4A080306
	v_add_nc_u32_e32 v12, 0x3f0, v11                           // 000000555104: 4A1816FF 000003F0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 00000055510C: BF8701A2
	v_ashrrev_i32_e32 v1, 3, v4                                // 000000555110: 34020883
	v_and_b32_e32 v7, -8, v4                                   // 000000555114: 360E08C8
	v_lshrrev_b32_e32 v9, 5, v12                               // 000000555118: 32121885
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000055511C: BF870193
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000555120: 3404029F
	v_sub_nc_u32_e32 v6, v6, v7                                // 000000555124: 4C0C0F06
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000555128: BF870193
	v_add_nc_u32_e32 v9, 1, v9                                 // 00000055512C: 4A121281
	v_mad_i64_i32 v[4:5], null, v3, s6, v[1:2]                 // 000000555130: D6FF7C04 04040D03
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000555138: BF870223
	v_ashrrev_i32_e32 v7, 31, v6                               // 00000055513C: 340E0C9F
	v_lshrrev_b32_e32 v2, 3, v12                               // 000000555140: 32041883
	v_and_b32_e32 v9, 31, v9                                   // 000000555144: 3612129F
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000555148: BF870193
	v_lshlrev_b64 v[7:8], 7, v[6:7]                            // 00000055514C: D73C0007 02020C87
	v_and_b32_e32 v2, 0x7c, v2                                 // 000000555154: 360404FF 0000007C
	v_lshlrev_b64 v[4:5], 10, v[4:5]                           // 00000055515C: D73C0004 0202088A
	s_waitcnt lgkmcnt(0)                                       // 000000555164: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000555168: BF870091
	v_add_co_u32 v4, vcc_lo, s2, v4                            // 00000055516C: D7006A04 02020802
	v_add_co_ci_u32_e64 v5, null, s3, v5, vcc_lo               // 000000555174: D5207C05 01AA0A03
	s_movk_i32 s2, 0x1eee                                      // 00000055517C: B0021EEE
	v_add_co_u32 v13, vcc_lo, v4, v7                           // 000000555180: D7006A0D 02020F04
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 000000555188: BF8701A1
	v_add_co_ci_u32_e64 v14, null, v5, v8, vcc_lo              // 00000055518C: D5207C0E 01AA1105
	v_lshlrev_b32_e32 v7, 2, v9                                // 000000555194: 300E1282
	v_add_co_u32 v4, vcc_lo, v13, v2                           // 000000555198: D7006A04 0202050D
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000005551A0: BF870191
	v_add_co_ci_u32_e64 v5, null, 0, v14, vcc_lo               // 0000005551A4: D5207C05 01AA1C80
	v_add_co_u32 v7, vcc_lo, v13, v7                           // 0000005551AC: D7006A07 02020F0D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000005551B4: BF8700A1
	v_add_co_ci_u32_e64 v8, null, 0, v14, vcc_lo               // 0000005551B8: D5207C08 01AA1C80
	v_add_co_u32 v9, vcc_lo, v13, v10                          // 0000005551C0: D7006A09 0202150D
	v_add_co_ci_u32_e64 v10, null, 0, v14, vcc_lo              // 0000005551C8: D5207C0A 01AA1C80
	s_clause 0x2                                               // 0000005551D0: BF850002
	global_load_b32 v5, v[4:5], off                            // 0000005551D4: DC520000 057C0004
	global_load_b32 v4, v[7:8], off                            // 0000005551DC: DC520000 047C0007
	global_load_b32 v9, v[9:10], off                           // 0000005551E4: DC520000 097C0009
	v_lshlrev_b32_e32 v2, 7, v3                                // 0000005551EC: 30040687
	v_mov_b16_e32 v10.h, 0                                     // 0000005551F0: 7F143880
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)// 0000005551F4: BF870232
	v_lshl_add_u32 v2, v6, 4, v2                               // 0000005551F8: D6460002 04090906
	v_and_b32_e32 v6, 0x7e0, v12                               // 000000555200: 360C18FF 000007E0
	v_and_b32_e32 v12, 16, v0                                  // 000000555208: 36180090
	v_mov_b16_e32 v14.h, v10.h                                 // 00000055520C: 7F1C398A
	v_mov_b16_e32 v15.h, v10.h                                 // 000000555210: 7F1E398A
	v_and_or_b32 v2, v0, 15, v2                                // 000000555214: D6570002 04091F00
	v_sub_nc_u32_e32 v6, v6, v11                               // 00000055521C: 4C0C1706
	v_lshlrev_b32_e32 v0, 4, v1                                // 000000555220: 30000284
	v_mov_b16_e32 v13.h, v10.h                                 // 000000555224: 7F1A398A
	v_mov_b16_e32 v11.h, v10.h                                 // 000000555228: 7F16398A
	v_mad_i64_i32 v[2:3], null, v2, s4, 0                      // 00000055522C: D6FF7C02 02000902
	v_add_nc_u32_e32 v7, 0xfffffc3c, v6                        // 000000555234: 4A0E0CFF FFFFFC3C
	v_add_nc_u32_e32 v8, 0xfffffc38, v6                        // 00000055523C: 4A100CFF FFFFFC38
	v_add_nc_u32_e32 v16, 0xfffffc34, v6                       // 000000555244: 4A200CFF FFFFFC34
	v_add_nc_u32_e32 v17, 0xfffffc30, v6                       // 00000055524C: 4A220CFF FFFFFC30
	v_ashrrev_i32_e32 v1, 31, v0                               // 000000555254: 3402009F
	v_lshlrev_b64 v[2:3], 1, v[2:3]                            // 000000555258: D73C0002 02020481
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000555260: BF870112
	v_lshlrev_b64 v[0:1], 1, v[0:1]                            // 000000555264: D73C0000 02020081
	v_add_co_u32 v18, vcc_lo, s0, v2                           // 00000055526C: D7006A12 02020400
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000555274: BF870111
	v_add_co_ci_u32_e64 v19, null, s1, v3, vcc_lo              // 000000555278: D5207C13 01AA0601
	v_add_co_u32 v0, vcc_lo, v18, v0                           // 000000555280: D7006A00 02020112
	s_delay_alu instid0(VALU_DEP_1)                            // 000000555288: BF870001
	v_add_co_ci_u32_e64 v1, null, v19, v1, vcc_lo              // 00000055528C: D5207C01 01AA0313
	s_waitcnt vmcnt(1)                                         // 000000555294: BF8907F7
	v_lshrrev_b64 v[2:3], v7, v[4:5]                           // 000000555298: D73D0002 02020907
	v_lshrrev_b64 v[6:7], v8, v[4:5]                           // 0000005552A0: D73D0006 02020908
	v_lshrrev_b64 v[7:8], v16, v[4:5]                          // 0000005552A8: D73D0007 02020910
	v_lshrrev_b64 v[3:4], v17, v[4:5]                          // 0000005552B0: D73D0003 02020911
	s_waitcnt vmcnt(0)                                         // 0000005552B8: BF8903F7
	v_bfe_u32 v4, v9, 12, 16                                   // 0000005552BC: D6100004 02411909
	v_bfe_u32 v8, v9, 4, 16                                    // 0000005552C4: D6100008 02410909
	v_and_b32_e32 v2, 0xffff, v2                               // 0000005552CC: 360404FF 0000FFFF
	v_and_b32_e32 v6, 0xffff, v6                               // 0000005552D4: 360C0CFF 0000FFFF
	v_and_b32_e32 v7, 0xffff, v7                               // 0000005552DC: 360E0EFF 0000FFFF
	v_and_b32_e32 v3, 0xffff, v3                               // 0000005552E4: 360606FF 0000FFFF
	v_bfe_u32 v5, v9, 8, 16                                    // 0000005552EC: D6100005 02411109
	v_mul_lo_u32 v2, 0x83dcd12d, v2                            // 0000005552F4: D72C0002 020204FF 83DCD12D
	v_mul_lo_u32 v6, 0x83dcd12d, v6                            // 000000555300: D72C0006 02020CFF 83DCD12D
	v_mul_lo_u32 v7, 0x83dcd12d, v7                            // 00000055530C: D72C0007 02020EFF 83DCD12D
	v_mul_lo_u32 v4, 0x83dcd12d, v4                            // 000000555318: D72C0004 020208FF 83DCD12D
	v_and_b32_e32 v9, 0xffff, v9                               // 000000555324: 361212FF 0000FFFF
	v_mul_lo_u32 v8, 0x83dcd12d, v8                            // 00000055532C: D72C0008 020210FF 83DCD12D
	v_mul_lo_u32 v3, 0x83dcd12d, v3                            // 000000555338: D72C0003 020206FF 83DCD12D
	v_mul_lo_u32 v5, 0x83dcd12d, v5                            // 000000555344: D72C0005 02020AFF 83DCD12D
	v_sad_u8 v2, v2, 0, 0x6400                                 // 000000555350: D6220002 03FD0102 00006400
	v_sad_u8 v6, v6, 0, 0x6400                                 // 00000055535C: D6220006 03FD0106 00006400
	v_sad_u8 v7, v7, 0, 0x6400                                 // 000000555368: D6220007 03FD0107 00006400
	v_mul_lo_u32 v9, 0x83dcd12d, v9                            // 000000555374: D72C0009 020212FF 83DCD12D
	v_sad_u8 v4, v4, 0, 0x6400                                 // 000000555380: D6220004 03FD0104 00006400
	v_and_b32_e32 v2, 0xffff, v2                               // 00000055538C: 360404FF 0000FFFF
	v_sad_u8 v8, v8, 0, 0x6400                                 // 000000555394: D6220008 03FD0108 00006400
	v_sad_u8 v3, v3, 0, 0x6400                                 // 0000005553A0: D6220003 03FD0103 00006400
	v_and_b32_e32 v7, 0xffff, v7                               // 0000005553AC: 360E0EFF 0000FFFF
	v_sad_u8 v5, v5, 0, 0x6400                                 // 0000005553B4: D6220005 03FD0105 00006400
	v_lshl_or_b32 v2, v6, 16, v2                               // 0000005553C0: D6560002 04092106
	v_and_b32_e32 v4, 0xffff, v4                               // 0000005553C8: 360808FF 0000FFFF
	v_sad_u8 v6, v9, 0, 0x6400                                 // 0000005553D0: D6220006 03FD0109 00006400
	v_and_b32_e32 v8, 0xffff, v8                               // 0000005553DC: 361010FF 0000FFFF
	v_lshl_or_b32 v3, v3, 16, v7                               // 0000005553E4: D6560003 041D2103
	v_pk_fma_f16 v2, v2, s2, 0xc931 op_sel_hi:[1,0,0]          // 0000005553EC: CC0E0002 0BFC0502 0000C931
	v_lshl_or_b32 v4, v5, 16, v4                               // 0000005553F8: D6560004 04112105
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000555400: BF870214
	v_lshl_or_b32 v6, v6, 16, v8                               // 000000555404: D6560006 04212106
	v_pk_fma_f16 v3, v3, s2, 0xc931 op_sel_hi:[1,0,0]          // 00000055540C: CC0E0003 0BFC0503 0000C931
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000555418: BF870214
	v_cvt_f32_f16_e32 v8, v2.l                                 // 00000055541C: 7E101702
	v_pk_fma_f16 v7, v4, s2, 0xc931 op_sel_hi:[1,0,0]          // 000000555420: CC0E0007 0BFC0504 0000C931
	v_add_co_u32 v4, vcc_lo, v0, v12                           // 00000055542C: D7006A04 02021900
	s_delay_alu instid0(VALU_DEP_1)                            // 000000555434: BF870001
	v_add_co_ci_u32_e64 v5, null, 0, v1, vcc_lo                // 000000555438: D5207C05 01AA0280
	v_pk_fma_f16 v0, v6, s2, 0xc931 op_sel_hi:[1,0,0]          // 000000555440: CC0E0000 0BFC0506 0000C931
	v_mov_b16_e32 v10.l, v8.h                                  // 00000055544C: 7E143988
	v_cvt_f32_f16_e32 v1, v3.l                                 // 000000555450: 7E021703
	v_cvt_f32_f16_e32 v6, v7.h                                 // 000000555454: 7E0C1787
	v_cvt_f32_f16_e32 v7, v7.l                                 // 000000555458: 7E0E1707
	v_cvt_f32_f16_e32 v9, v0.h                                 // 00000055545C: 7E121780
	v_and_b32_e32 v12, 1, v10                                  // 000000555460: 36181481
	v_mov_b16_e32 v10.l, v1.h                                  // 000000555464: 7E143981
	v_cvt_f32_f16_e32 v16, v0.l                                // 000000555468: 7E201700
	v_mov_b16_e32 v14.l, v6.h                                  // 00000055546C: 7E1C3986
	v_mov_b16_e32 v15.l, v9.h                                  // 000000555470: 7E1E3989
	v_cvt_f32_f16_e32 v0, v2.h                                 // 000000555474: 7E001782
	v_and_b32_e32 v17, 1, v10                                  // 000000555478: 36221481
	v_mov_b16_e32 v10.l, v7.h                                  // 00000055547C: 7E143987
	v_cvt_f32_f16_e32 v18, v3.h                                // 000000555480: 7E241783
	v_and_b32_e32 v2, 1, v14                                   // 000000555484: 36041C81
	v_and_b32_e32 v3, 1, v15                                   // 000000555488: 36061E81
	v_mov_b16_e32 v13.l, v0.h                                  // 00000055548C: 7E1A3980
	v_and_b32_e32 v14, 1, v10                                  // 000000555490: 361C1481
	v_mov_b16_e32 v10.l, v16.h                                 // 000000555494: 7E143990
	v_mov_b16_e32 v11.l, v18.h                                 // 000000555498: 7E163992
	v_add3_u32 v2, v6, v2, 0x7fff                              // 00000055549C: D6550002 03FE0506 00007FFF
	v_and_b32_e32 v6, 1, v13                                   // 0000005554A8: 360C1A81
	v_add3_u32 v3, v9, v3, 0x7fff                              // 0000005554AC: D6550003 03FE0709 00007FFF
	v_and_b32_e32 v9, 1, v10                                   // 0000005554B8: 36121481
	v_and_b32_e32 v10, 1, v11                                  // 0000005554BC: 36141681
	v_add3_u32 v8, v8, v12, 0x7fff                             // 0000005554C0: D6550008 03FE1908 00007FFF
	v_add3_u32 v0, v0, v6, 0x7fff                              // 0000005554CC: D6550000 03FE0D00 00007FFF
	v_add3_u32 v6, v1, v17, 0x7fff                             // 0000005554D8: D6550006 03FE2301 00007FFF
	v_add3_u32 v9, v16, v9, 0x7fff                             // 0000005554E4: D6550009 03FE1310 00007FFF
	v_add3_u32 v7, v7, v14, 0x7fff                             // 0000005554F0: D6550007 03FE1D07 00007FFF
	v_add3_u32 v1, v18, v10, 0x7fff                            // 0000005554FC: D6550001 03FE1512 00007FFF
	v_mov_b16_e32 v0.l, v8.h                                   // 000000555508: 7E003988
	v_mov_b16_e32 v1.l, v6.h                                   // 00000055550C: 7E023986
	v_mov_b16_e32 v3.l, v9.h                                   // 000000555510: 7E063989
	v_mov_b16_e32 v2.l, v7.h                                   // 000000555514: 7E043987
	global_store_b128 v[4:5], v[0:3], off                      // 000000555518: DC760000 007C0004
	s_endpgm                                                   // 000000555520: BFB00000
		...
