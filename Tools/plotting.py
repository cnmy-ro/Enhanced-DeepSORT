import matplotlib.pyplot as plt

'''
All values are copied from the benchmark result files

'''


###########################################################################
# Cosine distance expt
###########################################################################


max_cosine_dist_list = [0.2, 0.3, 0.4, 0.5]


# Pedestrian tracking ------------------------------------------------------------------
ped_mota_dpm_list = [72.6, 70.0, 75.3, 75.4]
ped_motp_dpm_list = [0.467, 0.486, 0.549, 0.540]
ped_id_dpm_list = [836, 919, 905, 939]
ped_fm_dpm_list = [1290, 1427, 1475, 1516]

ped_mota_ssd_list = [24.5, 25.4, 25.9, 26.1]
ped_motp_ssd_list = [0.601, 0.634, 0.678, 0.685]
ped_id_ssd_list = [644, 569, 499, 500]
ped_fm_ssd_list = [1607, 1687, 1773, 1807]


fig, axs = plt.subplots(1,2) # Plotting MOTA and MOTP
fig.set_size_inches(0.6*18.5, 0.4*10.5)
fig.subplots_adjust(wspace=0.3)
fig.set_dpi(150)
ax_1 = axs[0]
ax_1.plot(max_cosine_dist_list, ped_mota_dpm_list, 'bo-', label="DPMv5")
ax_1.plot(max_cosine_dist_list, ped_mota_ssd_list, 'ro-', label="MobileNetv2-SSDLite")
ax_1.set_xticks(max_cosine_dist_list)
ax_1.set_xlabel("Max Cosine Distance")
ax_1.set_ylabel("MOTA score")
ax_1.set_title("MOTA v/s Cosine Distance Limit")
ax_1.legend()
ax_1.grid()

ax_2 = axs[1]
ax_2.plot(max_cosine_dist_list, ped_motp_dpm_list, 'co-', label="DPMv5")
ax_2.plot(max_cosine_dist_list, ped_motp_ssd_list, 'mo-', label="MobileNetv2-SSDLite")
ax_2.set_xticks(max_cosine_dist_list)
ax_2.set_xlabel("Max Cosine Distance")
ax_2.set_ylabel("MOTP score")
ax_2.set_title("MOTP v/s Cosine Distance Limit")
ax_2.legend()
ax_2.grid()

fig.savefig("Results/Pedestrian Tracking/ped_cosineThresh_motp_mota.png")
plt.show()



fig, axs = plt.subplots(1,2) # Plotting ID and FM
fig.set_size_inches(0.6*18.5, 0.4*10.5)
fig.subplots_adjust(wspace=0.3)
fig.set_dpi(150)
ax_3 = axs[0]
ax_3.plot(max_cosine_dist_list, ped_id_dpm_list, 'o-', color= 'DarkSlateBlue', label="DPMv5")
ax_3.plot(max_cosine_dist_list, ped_id_ssd_list, 'o-', color= 'Maroon', label="MobileNetv2-SSDLite")
ax_3.set_xticks(max_cosine_dist_list)
ax_3.set_xlabel("Max Cosine Distance")
ax_3.set_ylabel("Number of Identity Switches")
ax_3.set_title("ID v/s Cosine Distance Limit")
ax_3.legend()
ax_3.grid()

ax_4 = axs[1]
ax_4.plot(max_cosine_dist_list, ped_fm_dpm_list, 'o-', color= 'BlueViolet', label="DPMv5")
ax_4.plot(max_cosine_dist_list, ped_fm_ssd_list, 'o-', color= 'Crimson', label="MobileNetv2-SSDLite")
ax_4.set_xticks(max_cosine_dist_list)
ax_4.set_xlabel("Max Cosine Distance")
ax_4.set_ylabel("Number of Track Fragmentations")
ax_4.set_title("FM v/s Cosine Distance Limit")
ax_4.legend()
ax_4.grid()

#fig.suptitle("Pedestrian Tracking", fontsize='x-large')
fig.savefig("Results/Pedestrian Tracking/ped_cosineThresh_id_fm.png")
plt.show()



# Vehicle tracking ------------------------------------------------------------------
veh_mota_dpm_list = [34.2, 34.2, 34.2, 34.2]
veh_motp_dpm_list = [0.589, 0.586, 0.574, 0.583]
veh_id_dpm_list = [991, 988, 978, 980]
veh_fm_dpm_list = [2570, 2576, 2578, 2572]

veh_mota_rcnn_list = [51.1, 51.3, 51.1, 51.0]
veh_motp_rcnn_list = [0.544, 0.540, 0.545, 0.541]
veh_id_rcnn_list = [5497, 5455, 5496, 5506]
veh_fm_rcnn_list = [1238, 1250, 1238, 1227]

veh_mota_ssd_list = [34.1, 34.0, 34.0, 34.1]
veh_motp_ssd_list = [0.830, 0.832, 0.828, 0.824]
veh_id_ssd_list = [6514, 6560, 6483, 6448]
veh_fm_ssd_list = [1098, 1101, 1101, 1095]


fig, axs = plt.subplots(1,2) # Plotting MOTA and MPTP
fig.set_size_inches(0.6*18.5, 0.4*10.5)
fig.set_dpi(150)
ax_1 = axs[0]
ax_1.plot(max_cosine_dist_list, veh_mota_dpm_list, 'bo-', label="DPM")
ax_1.plot(max_cosine_dist_list, veh_mota_rcnn_list, 'go-', label="R-CNN")
ax_1.plot(max_cosine_dist_list, veh_mota_ssd_list, 'ro-', label="MobileNetv2-SSDLite")
ax_1.set_xticks(max_cosine_dist_list)
ax_1.set_xlabel("Max Cosine Distance")
ax_1.set_ylabel("MOTA score")
ax_1.set_title("MOTA v/s Cosine Distance Limit")
ax_1.legend()
ax_1.grid()

ax_2 = axs[1]
ax_2.plot(max_cosine_dist_list, veh_motp_dpm_list, 'co-', label="DPM")
ax_2.plot(max_cosine_dist_list, veh_motp_rcnn_list, 'yo-', label="R-CNN")
ax_2.plot(max_cosine_dist_list, veh_motp_ssd_list, 'mo-', label="MobileNetv2-SSDLite")
ax_2.set_xticks(max_cosine_dist_list)
ax_2.set_xlabel("Max Cosine Distance")
ax_2.set_ylabel("MOTP score")
ax_2.set_title("MOTP v/s Cosine Distance Limit")
ax_2.legend()
ax_2.grid()


#fig.suptitle("Vehicle Tracking", fontsize='x-large')
fig.savefig("Results/Vehicle Tracking/veh_cosineThresh_mota_motp.png")
plt.show()


fig, axs = plt.subplots(1,2) # Plotting ID and FM
fig.set_size_inches(0.6*18.5, 0.4*10.5)
fig.subplots_adjust(wspace=0.3)
fig.set_dpi(150)
ax_3 = axs[0]
ax_3.plot(max_cosine_dist_list, veh_id_dpm_list, 'o-', color= 'DarkSlateBlue', label="DPM")
ax_3.plot(max_cosine_dist_list, veh_id_rcnn_list, 'o-', color= 'SeaGreen', label="R-CNN")
ax_3.plot(max_cosine_dist_list, veh_id_ssd_list, 'o-', color= 'Maroon', label="MobileNetv2-SSDLite")
ax_3.set_xticks(max_cosine_dist_list)
ax_3.set_xlabel("Max Cosine Distance")
ax_3.set_ylabel("Number of Identity Switches")
ax_3.set_title("ID v/s Cosine Distance Limit")
ax_3.legend()
ax_3.grid()

ax_4 = axs[1]
ax_4.plot(max_cosine_dist_list, veh_fm_dpm_list, 'o-', color= 'BlueViolet', label="DPM")
ax_4.plot(max_cosine_dist_list, veh_fm_rcnn_list, 'o-', color= 'GreenYellow', label="R-CNN")
ax_4.plot(max_cosine_dist_list, veh_fm_ssd_list, 'o-', color= 'Crimson', label="MobileNetv2-SSDLite")
ax_4.set_xticks(max_cosine_dist_list)
ax_4.set_xlabel("Max Cosine Distance")
ax_4.set_ylabel("Number of Track Fragmentations")
ax_4.set_title("FM v/s Cosine Distance Limit")
ax_4.legend()
ax_4.grid()

fig.savefig("Results/Vehicle Tracking/veh_cosineThresh_id_fm.png")
plt.show()














###########################################################################
# Min confidence expt
###########################################################################


min_conf_list = [0.0, 0.2, 0.4, 0.6]


# Pedestrian tracking ------------------------------------------------------------------
ped_mota_dpm_list = [80.3, 78.3, 72.6, 71.1]
ped_motp_dpm_list = [0.568, 0.486, 0.467, 0.429]
ped_id_dpm_list = [1909, 1330, 836, 754]
ped_fm_dpm_list = [1445, 1420, 1290, 1249]

ped_mota_ssd_list = [24.7, 24.7, 24.5, 24.7]
ped_motp_ssd_list = [0.601, 0.601, 0.601, 0.601]
ped_id_ssd_list = [644, 644, 644, 644]
ped_fm_ssd_list = [1593, 1593, 1607, 1593]


fig, axs = plt.subplots(1,2) # Plotting MOTA and MOTP
fig.set_size_inches(0.6*18.5, 0.4*10.5)
fig.subplots_adjust(wspace=0.3)
fig.set_dpi(150)
ax_1 = axs[0]
ax_1.plot(min_conf_list, ped_mota_dpm_list, 'bo-', label="DPMv5")
ax_1.plot(min_conf_list, ped_mota_ssd_list, 'ro-', label="MobileNetv2-SSDLite")
ax_1.set_xticks(min_conf_list)
ax_1.set_xlabel("Minimum Confidence")
ax_1.set_ylabel("MOTA score")
ax_1.set_title("MOTA v/s Minimum Confidence Threshold")
ax_1.legend()
ax_1.grid()

ax_2 = axs[1]
ax_2.plot(min_conf_list, ped_motp_dpm_list, 'co-', label="DPMv5")
ax_2.plot(min_conf_list, ped_motp_ssd_list, 'mo-', label="MobileNetv2-SSDLite")
ax_2.set_xticks(min_conf_list)
ax_2.set_xlabel("Minimum Confidence")
ax_2.set_ylabel("MOTP score")
ax_2.set_title("MOTP v/s Minimum Confidence Threshold")
ax_2.legend()
ax_2.grid()

fig.savefig("Results/Pedestrian Tracking/ped_minConf_motp_mota.png")
plt.show()


fig, axs = plt.subplots(1,2) # Plotting ID and FM
fig.set_size_inches(0.6*18.5, 0.4*10.5)
fig.subplots_adjust(wspace=0.3)
fig.set_dpi(150)
ax_3 = axs[0]
ax_3.plot(min_conf_list, ped_id_dpm_list, 'o-', color= 'DarkSlateBlue', label="DPMv5")
ax_3.plot(min_conf_list, ped_id_ssd_list, 'o-', color= 'Maroon', label="MobileNetv2-SSDLite")
ax_3.set_xticks(min_conf_list)
ax_3.set_xlabel("Minimum Confidence")
ax_3.set_ylabel("Number of Identity Switches")
ax_3.set_title("ID v/s Minimum Confidence Threshold")
ax_3.legend()
ax_3.grid()

ax_4 = axs[1]
ax_4.plot(min_conf_list, ped_fm_dpm_list, 'o-', color= 'BlueViolet', label="DPMv5")
ax_4.plot(min_conf_list, ped_fm_ssd_list, 'o-', color= 'Crimson', label="MobileNetv2-SSDLite")
ax_4.set_xticks(min_conf_list)
ax_4.set_xlabel("Minimum Confidence")
ax_4.set_ylabel("Number of Track Fragmentations")
ax_4.set_title("FM v/s Minimum Confidence Threshold")
ax_4.legend()
ax_4.grid()

#fig.suptitle("Pedestrian Tracking", fontsize='x-large')
fig.savefig("Results/Pedestrian Tracking/ped_minConf_id_fm.png")
plt.show()



# Vehicle tracking ------------------------------------------------------------------

min_conf_list = [0.1, 0.2, 0.3, 0.4]


veh_mota_dpm_list = [45.9, 66.5, 49.8, 34.2]
veh_motp_dpm_list = [0.742, 0.786, 0.718, 0.589]
veh_id_dpm_list = [7874, 4889, 2174, 991]
veh_fm_dpm_list = [1934, 3859, 3832, 2570]

veh_mota_rcnn_list = [42.6, 42.6, 43.3, 51.1]
veh_motp_rcnn_list = [0.504, 0.504, 0.523, 0.544]
veh_id_rcnn_list = [5548, 5572, 5625, 5497]
veh_fm_rcnn_list = [1106, 1106, 1118, 1238]

veh_mota_ssd_list = [28.0, 28.0, 28.0, 34.1]
veh_motp_ssd_list = [0.763, 0.763, 0.763, 0.830]
veh_id_ssd_list = [6393, 6393, 6393, 6514]
veh_fm_ssd_list = [321, 321, 321, 1098]


fig, axs = plt.subplots(1,2) # Plotting MOTA and MPTP
fig.set_size_inches(0.6*18.5, 0.4*10.5)
fig.set_dpi(150)
ax_1 = axs[0]
ax_1.plot(min_conf_list, veh_mota_dpm_list, 'bo-', label="DPM")
ax_1.plot(min_conf_list, veh_mota_rcnn_list, 'go-', label="R-CNN")
ax_1.plot(min_conf_list, veh_mota_ssd_list, 'ro-', label="MobileNetv2-SSDLite")
ax_1.set_xticks(min_conf_list)
ax_1.set_xlabel("Minimum Confidence")
ax_1.set_ylabel("MOTA score")
ax_1.set_title("MOTA v/s Minimum Confidence Threshold")
ax_1.legend()
ax_1.grid()

ax_2 = axs[1]
ax_2.plot(min_conf_list, veh_motp_dpm_list, 'co-', label="DPM")
ax_2.plot(min_conf_list, veh_motp_rcnn_list, 'yo-', label="R-CNN")
ax_2.plot(min_conf_list, veh_motp_ssd_list, 'mo-', label="MobileNetv2-SSDLite")
ax_2.set_xticks(min_conf_list)
ax_2.set_xlabel("Minimum Confidence")
ax_2.set_ylabel("MOTP score")
ax_2.set_title("MOTP v/s Minimum Confidence Threshold")
ax_2.legend()
ax_2.grid()

#fig.suptitle("Vehicle Tracking", fontsize='x-large')
fig.savefig("Results/Vehicle Tracking/veh_minConf_mota_motp.png")
plt.show()


fig, axs = plt.subplots(1,2) # Plotting ID and FM
fig.set_size_inches(0.6*18.5, 0.4*10.5)
fig.subplots_adjust(wspace=0.3)
fig.set_dpi(150)
ax_3 = axs[0]
ax_3.plot(min_conf_list, veh_id_dpm_list, 'o-', color= 'DarkSlateBlue', label="DPM")
ax_3.plot(min_conf_list, veh_id_rcnn_list, 'o-', color= 'SeaGreen', label="R-CNN")
ax_3.plot(min_conf_list, veh_id_ssd_list, 'o-', color= 'Maroon', label="MobileNetv2-SSDLite")
ax_3.set_xticks(min_conf_list)
ax_3.set_xlabel("Minimum Confidence")
ax_3.set_ylabel("Number of Identity Switches")
ax_3.set_title("ID v/s Minimum Confidence Threshold")
ax_3.legend()
ax_3.grid()

ax_4 = axs[1]
ax_4.plot(min_conf_list, veh_fm_dpm_list, 'o-', color= 'BlueViolet', label="DPM")
ax_4.plot(min_conf_list, veh_fm_rcnn_list, 'o-', color= 'GreenYellow', label="R-CNN")
ax_4.plot(min_conf_list, veh_fm_ssd_list, 'o-', color= 'Crimson', label="MobileNetv2-SSDLite")
ax_4.set_xticks(min_conf_list)
ax_4.set_xlabel("Minimum Confidence")
ax_4.set_ylabel("Number of Track Fragmentations")
ax_4.set_title("FM v/s Minimum Confidence Threshold")
ax_4.legend()
ax_4.grid()

fig.savefig("Results/Vehicle Tracking/veh_minConf_id_fm.png")
plt.show()