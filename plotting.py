import matplotlib.pyplot as plt


max_cosine_dist_list = [0.2, 0.3, 0.4, 0.5]


# Pedestrian tracking
ped_mota_dpm_list = [72.6, 70.0, 75.3, 75.4]
ped_motp_dpm_list = [0.467, 0.486, 0.549, 0.540]

ped_mota_ssd_list = [24.5, 25.4, 25.9, 26.1]
ped_motp_ssd_list = [0.601, 0.634, 0.678, 0.685]


fig, axs = plt.subplots(1,2)
fig.set_size_inches(0.5*18.5, 0.4*10.5)
fig.set_dpi(150)
ax_1 = axs[0]
ax_1.plot(max_cosine_dist_list, ped_mota_dpm_list, 'bo-', label="DPMv5")
ax_1.plot(max_cosine_dist_list, ped_mota_ssd_list, 'ro-', label="MobileNetv2-SSD")
ax_1.set_xticks(max_cosine_dist_list)
ax_1.set_xlabel("Max Cosine Distance")
ax_1.set_ylabel("MOTA score")
ax_1.set_title("MOTA v/s Cosine Distance Limit")
ax_1.legend()

ax_2 = axs[1]
ax_2.plot(max_cosine_dist_list, ped_motp_dpm_list, 'co-', label="DPMv5")
ax_2.plot(max_cosine_dist_list, ped_motp_ssd_list, 'mo-', label="MobileNetv2-SSD")
ax_2.set_xticks(max_cosine_dist_list)
ax_2.set_xlabel("Max Cosine Distance")
ax_2.set_ylabel("MOTP score")
ax_2.set_title("MOTP v/s Cosine Distance Limit")
ax_2.legend()

fig.suptitle("Pedestrian Tracking", fontsize='x-large')
fig.savefig("plot_pedestrians.png")
plt.show()


# Vehicle tracking
veh_mota_dpm_list = [34.2, 34.2, 34.2, 34.2]
veh_mota_rcnn_list = [51.1, 51.3, 51.1, 51.0]
veh_mota_ssd_list = [34.1, 34.0, 34.0, 34.1]

veh_motp_dpm_list = [0.589, 0.586, 0.574, 0.583]
veh_motp_rcnn_list = [0.544, 0.540, 0.545, 0.541]
veh_motp_ssd_list = [0.830, 0.832, 0.828, 0.824]


fig, axs = plt.subplots(1,2)
fig.set_size_inches(0.5*18.5, 0.4*10.5)
fig.set_dpi(150)
ax_1 = axs[0]
ax_1.plot(max_cosine_dist_list, veh_mota_dpm_list, 'bo-', label="DPM")
ax_1.plot(max_cosine_dist_list, veh_mota_rcnn_list, 'go-', label="R-CNN")
ax_1.plot(max_cosine_dist_list, veh_mota_ssd_list, 'ro-', label="MobileNetv2-SSD")
ax_1.set_xticks(max_cosine_dist_list)
ax_1.set_xlabel("Max Cosine Distance")
ax_1.set_ylabel("MOTA score")
ax_1.set_title("MOTA v/s Cosine Distance Limit")
ax_1.legend()

ax_2 = axs[1]
ax_2.plot(max_cosine_dist_list, veh_motp_dpm_list, 'co-', label="DPM")
ax_2.plot(max_cosine_dist_list, veh_motp_rcnn_list, 'yo-', label="R-CNN")
ax_2.plot(max_cosine_dist_list, veh_motp_ssd_list, 'mo-', label="MobileNetv2-SSD")
ax_2.set_xticks(max_cosine_dist_list)
ax_2.set_xlabel("Max Cosine Distance")
ax_2.set_ylabel("MOTP score")
ax_2.set_title("MOTP v/s Cosine Distance Limit")
ax_2.legend()

fig.suptitle("Vehicle Tracking", fontsize='x-large')
fig.savefig("plot_vehicles.png")
plt.show()