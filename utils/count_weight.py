import numpy
import nibabel as nib


path = 'data/data_2.1/Seg/sub-001_dseg.nii.gz'
a = nib.load(path).get_fdata()

# a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
unique, counts = numpy.unique(a, return_counts=True)
prec = counts/(180*180*180)
print(prec)
print(dict(zip(unique, prec)))