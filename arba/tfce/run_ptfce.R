#!/usr/bin/env Rscript
library("optparse")
library("pTFCE")
library("dcemriS4")

option_list = list(
make_option(c("-i", "--fin"), type = "character", default = NULL, help = "input nifti", metavar = "character"),
make_option(c("-o", "--out"), type = "character", default = NULL, help = "output nifti", metavar = "character"),
make_option(c("-m", "--mask"), type = "character", default = NULL, help = "mask nifti", metavar = "character"),
make_option(c("-v", "--voxels"), type = "numeric", default = NULL, help = "total volume", metavar = "numeric"),
make_option(c("-r", "--resels"), type = "numeric", default = NULL, help = "resels", metavar = "numeric")
);

opt_parser = OptionParser(option_list = option_list);
opt = parse_args(opt_parser);

z = readNIfTI(opt$fin)
mask = readNIfTI(opt$mask)
pTFCE = ptfce(z, mask, Rd=opt$resels, V=opt$voxels)

writeNIfTI(pTFCE$Z, opt$out)