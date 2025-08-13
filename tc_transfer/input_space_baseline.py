import numpy as np
from cesnet_datazoo.constants import DIR_POS, IPT_POS, SIZE_POS

DEFAULT_BASELINE_PARAMS = {
    "num_packets": 10,
    "small_packet_threshold": 0,
    "dir_scale": 1,
    "ipt_max_clip": 1000, # ms
    "ipt_scale": 0.1,
}

def prepare_input_space_embeddings(data, baseline_params: dict):
    num_packets = baseline_params.get("num_packets", DEFAULT_BASELINE_PARAMS["num_packets"])
    small_packet_threshold = baseline_params.get("small_packet_threshold", DEFAULT_BASELINE_PARAMS["small_packet_threshold"])
    dir_scale = baseline_params.get("dir_scale", DEFAULT_BASELINE_PARAMS["dir_scale"])
    ipt_max_clip = baseline_params.get("ip_max_clip", DEFAULT_BASELINE_PARAMS["ipt_max_clip"])
    ipt_scale = baseline_params.get("ipt_scale", DEFAULT_BASELINE_PARAMS["ipt_scale"])

    data = data[:, :, :num_packets]
    sizes = data[:, SIZE_POS].clip(min=0, max=1500)
    dirs = data[:, DIR_POS] * dir_scale
    times = data[:, IPT_POS].clip(min=0, max=ipt_max_clip) * ipt_scale

    # Filtering out small packets if configured
    if small_packet_threshold != 0:
        embedding_list = []
        for i in range(sizes.shape[0]):
            keep_mask = sizes[i] > small_packet_threshold
            s = sizes[i][keep_mask]
            d = dirs[i][keep_mask]
            t = times[i][keep_mask]
            padded = np.pad((d, s, t), pad_width=((0, 0), (0, num_packets - keep_mask.sum())))
            embedding_list.append(padded.flatten())
        embeddings = np.stack(embedding_list)
    else:
        embeddings =  np.hstack((dirs, sizes, times))
    return embeddings
