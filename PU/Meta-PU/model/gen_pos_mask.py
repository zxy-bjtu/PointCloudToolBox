
import math
import torch
import numpy as np


scale = np.arange(1.1,4.1,0.1) # a loop of scale from 1.1 to 4

######by given the scale and the size of input image
######we caculate the input matrix for the weight prediction network
def input_matrix_wpn(inN, outN, scale, add_scale=True):
    '''
    inN: the size of the feature maps
    scale: is the upsampling times
    '''
    # outN = int(scale * inN)
    #### mask records which pixel is invalid, 1 valid or o invalid
    #### h_offset and w_offset caculate the offset to generate the input matrix
    scale_int = int(math.ceil(scale))
    offset = torch.ones(inN, scale_int, 1)
    mask = torch.zeros(inN, scale_int, 1)

    if add_scale:
        # scale_mat = torch.zeros(1)
        # scale_mat[0] = 1.0 / scale
        scale_mat = torch.zeros(1, 1)
        scale_mat[0, 0] = 1.0 / scale
        # res_scale = scale_int - scale
        # scale_mat[0,scale_int-1]=1-res_scale
        # scale_mat[0,scale_int-2]= res_scale
        scale_mat = torch.cat([scale_mat] * (inN * (scale_int)), 0)  ###(inH*inW*scale_int**2, 4)

    ####projection  coordinate  and caculate the offset
    project_coord = torch.arange(0, outN, 1).float().mul(1.0 / scale)  # i/r
    int_project_coord = torch.floor(project_coord)  # floor(i/r)

    offset_coord = project_coord - int_project_coord  # R(i)
    int_project_coord = int_project_coord.int()

    ####flag for number for current coordinate LR image
    flag = 0
    number = 0
    for i in range(outN):
        if int_project_coord[i] == number:
            offset[int_project_coord[i], flag, 0] = offset_coord[i]
            mask[int_project_coord[i], flag, 0] = 1
            flag += 1
        else:
            offset[int_project_coord[i], 0, 0] = offset_coord[i]
            mask[int_project_coord[i], 0, 0] = 1
            number += 1
            flag = 1

    ## the size is scale_int* inH* (scal_int*inW)
    pos_mat = offset.view(scale_int * inN, 1)
    # pos_mat = torch.cat((pos_mat, pos_mat), 2)
    # if self.multi_gpus:
    #     mask_mat = torch.nn.DataParallel(mask.view(-1, scale_int * inN, 1)).to(self.device)
    # else:
    #     mask_mat = mask.view(-1, scale_int * inN, 1).to(self.device)
    mask_mat = mask.view(-1, scale_int * inN, 1)
    pos_mat = pos_mat.contiguous().view(1, -1, 1)
    pos_mat = torch.cat((pos_mat,pos_mat),2)
    # if add_scale:
    #     if self.multi_gpus:
    #         pos_mat = torch.nn.DataParallel(torch.cat((scale_mat.view(1, -1, 1), pos_mat), 2)).to(self.device)
    #     else:
    #         pos_mat = torch.cat((scale_mat.view(1, -1, 1), pos_mat), 2).to(self.device)
    pos_mat = torch.cat((scale_mat.view(1, -1, 1), pos_mat), 2)
    return pos_mat, mask_mat.byte()


if __name__ == "__main__":
    save_f = []
    for this_scale in scale:
        gt_data_shape = 4096
        npoint_to_sample = int(gt_data_shape / this_scale)
        gt_npoint_to_sample = int(npoint_to_sample * this_scale)
        inN = npoint_to_sample
        outN2 = gt_npoint_to_sample

        first_up_scale = round(math.sqrt(this_scale), 3)  # baoliu sanwei xiaoshu
        outN1 = int(inN * first_up_scale)
        scale_coord_map1, mask1 = input_matrix_wpn(inN, outN1, first_up_scale)  ###  get the position matrix, mask
        second_up_scale = float(outN2) / float(outN1)
        scale_coord_map2, mask2 = input_matrix_wpn(outN1, outN2, second_up_scale)  ###  get the position matrix, mask

        state = {'scale_coord_map1': scale_coord_map1, 'mask1':mask1, 'scale_coord_map2': scale_coord_map2,
                 'mask2': mask2}
        save_f.append(state)
    torch.save(save_f,'pos_and_mask.pt')
