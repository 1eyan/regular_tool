# info_h5 = '/NAS/lyf/mount/TiT/Dataset_denoise/dataset_group_info.h5'

# segyPairs = {
#     '0': [
#         '/NAS/lyf/mount/Denoise_pretrain/Dataset/C_sht5_stap_NF_swt_CohN.sgy',
#         '/NAS/lyf/mount/Denoise_pretrain/Dataset/C_sht5_stap_NF_swt_CohN_denoise.sgy',
#         'denoise',
#         '5d_kdtree'
#     ],
#     '1': [
#         '/NAS/lyf/mount/Denoise_pretrain/Dataset/D_sht5_stap_NF_swt_CohN.sgy',
#         '/NAS/lyf/mount/Denoise_pretrain/Dataset/D_sht5_stap_NF_swt_CohN_eigen.sgy',
#         'denoise',
#         '5d_kdtree'
#     ],
#     '2': [
#         '/NAS/data/data/jiangyr/segy/001_raw_DX004_p2.sgy',
#         '/NAS/data/data/jiangyr/segy/006_3a3_nucns_3a2_data_DX004_p2.sgy',
#         'denoise',
#         '5d_kdtree'
#     ]
# }



# info_h5 = '/NAS/lyf/mount/TiT/Dataset_recon/dataset_group_info.h5'

# segyPairs = {
#     '0': [
#         '/NAS/lyf/mount/Data_cmp/cmp_sorted_all/reg5dbin_CMP_rawdata.sgy',
#         '/NAS/lyf/mount/Data_cmp/cmp_sorted_all/reg5dbin_CMP_Truelabel.sgy',
#         'interp',
#         '5d_kdtree'
#     ]
# }

# info_h5 = '/NAS/lyf/mount/TiT/Dataset_interp/dataset_group_info.h5'

# segyPairs = {
#     '0': [
#         '/NAS/lyf/mount/h5_to_segy/marmousi_5d.segy',
#         '/NAS/lyf/mount/h5_to_segy/marmousi_5d.segy',
#         'interp',
#         '5d_kdtree'
#     ]
# }

#info_h5 = '/NAS/czt/mount/seis_flow_data12V2/h5/dongfang/raw5d_data1104.h5'
info_h5 = './h5/segc3na/segc3na_2.h5'
segyPairs = {
    '1551': [
        '/data/shared/SEGC3/SEG_C3NA_ffid_1201-2400.sgy',
        '/data/shared/SEGC3/SEG_C3NA_ffid_1201-2400.sgy',
        'interp',
        '5d_line_by_order',
        'none'
    ]
}