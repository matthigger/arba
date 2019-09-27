from scipy.ndimage import binary_erosion

import arba


def sample_masks(effect_num_vox, data_img, num_eff=1, min_var_mask=False):
    """ gets list of masks (extent of effects) """

    assert data_img.is_loaded, 'file tree must be loaded'
    prior_array = data_img.mask

    # erode prior array if edges of prior_array are invalid (TFCE error)
    prior_array = binary_erosion(prior_array)

    # sample effect extent
    mask_list = list()
    for idx in range(num_eff):
        if min_var_mask:
            mask = arba.space.sample_mask_min_var(num_vox=effect_num_vox,
                                                  data_img=data_img,
                                                  prior_array=prior_array)
        else:
            mask = arba.space.sample_mask_cube(prior_array=prior_array,
                                               num_vox=effect_num_vox,
                                               ref=data_img.ref)
        mask_list.append(mask)
    return mask_list