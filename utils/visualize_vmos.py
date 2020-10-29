import numpy as np

unique_colors = np.array([\
    [0,0,0],\
    [128,0,0],\
    [0,128,0],\
    [0,0,128],\
    [128,128,0],\
    [128,0,128],\
    [0,128,128],\
    [255,0,0],\
    [0,255,0],\
    [0,0,255],\
    [255,255,0],\
    [255,0,255],\
    [0,255,255]])

def vis_vmos_maps(vmos_maps):

    img = np.zeros((vmos_maps.shape[0], vmos_maps.shape[1], 3))
    n_objects = vmos_maps.shape[2]
    for obj_idx in range(n_objects):
        obj_map = vmos_maps[:,:,obj_idx]
        obj_map = obj_map[:,:,None]
        obj_map = np.repeat(obj_map, 3, 2)
        color_values = unique_colors[obj_idx, :][None, None, :]
        img = img + obj_map * color_values
    return img 
    