#!/bin/bash


python3 test.py \
	-i $HOME/Temp/fivek/MIT_exposure/testraw/ \
	-e $HOME/Temp/fivek/MIT_exposure/testCsrgb/ \
	-l $HOME/Temp/fivek/MIT_exposure/val.txt \
	-md models/rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_test-94_best.pth \
	-np 10 \
	-nf 8 \
	-cs srgb \
	-at rgb \
	-od output_rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_test-94_best

 # optional arguments:
 #  -h, --help            show this help message and exit
 #  -i INPUT_DIR, --input_dir INPUT_DIR
 #                        The input dir containing the raw images. (default:
 #                        /media/flavio/Volume/datasets/fivek/raw/)
 #  -e EXPERTS_DIR [EXPERTS_DIR ...], --experts_dir EXPERTS_DIR [EXPERTS_DIR ...]
 #                        The experts dirs containing the gt. Can be more then
 #                        one. (default:
 #                        ['/media/flavio/Volume/datasets/fivek/ExpertC/'])
 #  -l TEST_LIST, --test_list TEST_LIST
 #                        File containing filenames. (default: /media/flavio/Vol
 #                        ume/datasets/fivek/test_mit_random250.txt)
 #  -md MODEL, --model MODEL
 #                        pth file containing the state dict of the model.
 #                        (default: )
 #  -np NPOINTS, --npoints NPOINTS
 #                        Number of points of the spline. (default: 10)
 #  -nf NFILTERS, --nfilters NFILTERS
 #                        Number of filters. (default: 32)
 #  -ds {maxpool,avgpool,convs,proj}, --downsample_strategy {maxpool,avgpool,convs,proj}
 #                        Type of downsampling. (default: avgpool)
 #  -bs BATCHSIZE, --batchsize BATCHSIZE
 #                        Batchsize. (default: 60)
 #  -cs {srgb,prophoto}, --colorspace {srgb,prophoto}
 #                        Colorspace to which belong images. (default: srgb)
 #  -at {rgb,lab}, --apply_to {rgb,lab}
 #                        Apply spline to rgb or lab images. (default: rgb)
 #  -abs, --abs           Applies absolute value to out rgb. (default: False)
 #  -de {76,94}, --deltae {76,94}
 #                        Version of the deltaE [76, 94]. (default: 94)
 #  -sem SEMSEG_DIR, --semseg_dir SEMSEG_DIR
 #                        Folder containing semantic segmentation. If empty,
 #                        model does not use semantic segmentation (default: )
 #  -nc NCLASSES, --nclasses NCLASSES
 #                        Number of classes of sem. seg. (default: 150)
 #  -sal SALIENCY_DIR, --saliency_dir SALIENCY_DIR
 #                        Folder containing semantic segmentation. If empty,
 #                        model does not use semantic segmentation (default: )
 #  -od OUT_DIR, --out_dir OUT_DIR
 #                        Output directory. (default: )
 #  -ods OUT_DIR_SPLINES, --out_dir_splines OUT_DIR_SPLINES
 #                        Output directory for splines. (default: )
