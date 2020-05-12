import os
from . import data_input
from .layered_ae import AE


def filter(cfg, points, p, p_bound, num_epoch=1000):
    data_obj = data_input.Data(points, p, p_bound)

    train_data_noisy, train_data_inter, train_data_denoised, npfactor = data_obj.get_training_data()

    network_arch = dict(encode1=64,
                        encode2=128,
                        encode3=256,
                        decode0=256,
                        decode1=128,
                        decode2=64,
                        latent_s=20,
                        input_size=len(train_data_noisy[0]),
                        output_size=len(train_data_denoised[0]))
    batch_s = min(512, len(train_data_noisy))
    ae = AE(network_arch, batch_s, num_epoch, os.path.join(cfg['project_path'], 'videos'))
    ae.train(train_data_noisy, train_data_inter, train_data_denoised)

    parallel_data, _ = data_obj.get_all_original_data()
    denoised_data = ae.reconstruct(parallel_data)

    return data_obj.reconstruct_original_data(denoised_data)

