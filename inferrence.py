
import numpy as np
import torch
import torch.nn.functional as F
from model_zoo import UNet
import nibabel as nib
import SimpleITK as sitk


def predict_img(net, full_img, device, out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(full_img)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        full_mask = probs.squeeze().cpu().numpy()
    mask = (full_mask > out_threshold)
    # mask = mask.transpose((1, 2, 3, 0))
    output = np.argmax(mask, axis=0)
    output = output.astype(np.int16)
    return output


if __name__ == "__main__":

    model = "./runs/test_64*128*128/unet_best_model.pth.tar"
    img_path = "data/imgs_crop/sub-015_T2w.nii.gz"

    net = UNet(n_channels=1, n_classes=8)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device)['state_dict'])

    img = nib.load(img_path)
    img_data = img.get_fdata()
    img_affine = img.affine
    img_data = img_data[np.newaxis, :]
     
    mask = predict_img(net=net,full_img=img_data,out_threshold=0.5,device=device)
    inference_path = "./data/sub-015_pred.nii.gz"
    predict_img = nib.Nifti1Image(mask, img_affine).to_filename(inference_path)

