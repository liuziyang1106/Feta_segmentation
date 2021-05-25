import os
import SimpleITK as sitk

def resampleVolume(outspacing,vol):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0,0,0]
    inputspacing = 0
    inputsize = 0
    inputorigin = [0,0,0]
    inputdir = [0,0,0]

    #读取文件的size和spacing信息
    
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()

    transform = sitk.Transform()
    transform.SetIdentity()
    #计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = int(inputsize[0]*inputspacing[0]/outspacing[0] + 0.5)
    outsize[1] = int(inputsize[1]*inputspacing[1]/outspacing[1] + 0.5)
    outsize[2] = int(inputsize[2]*inputspacing[2]/outspacing[2] + 0.5)

    #设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol

def resample_single_file():
    #读文件
    vol = sitk.Image(sitk.ReadImage("/home/liuziyang/workspace/FeTA/Pytorch-UNet/data/imgs/sub-041_rec-irtk_T2w.nii.gz"))
    #重采样
    newvol = resampleVolume([0.5,0.5,0.5],vol)
    #写文件
    wriiter = sitk.ImageFileWriter()
    wriiter.SetFileName("./output2.nii.gz")
    wriiter.Execute(newvol)

def resample_folder(folder_path, target_path, spacing=[0.5,0.5,0.5]):

    for file in os.listdir(folder_path):
        #读文件
        print(file)
        vol = sitk.Image(sitk.ReadImage(os.path.join(folder_path, file)))
        #重采样
        newvol = resampleVolume(spacing,vol)
        #写文件
        wriiter = sitk.ImageFileWriter()
        wriiter.SetFileName(os.path.join(target_path, file))
        wriiter.Execute(newvol)

resample_folder("/home/liuziyang/workspace/FeTA/Pytorch-UNet/data/masks/"
              , "/home/liuziyang/workspace/FeTA/Pytorch-UNet/data/mask_resample/")