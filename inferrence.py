
import numpy as np
import torch
import torch.nn.functional as F


from unet import UNet


import SimpleITK as sitk


def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(full_img)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


if __name__ == "__main__":

    model = "checkpoints_64128128/CP_epoch15.pth"
    img_path = "data/imgs_crop/sub-015_T2w.nii.gz"

    net = UNet(n_channels=1, n_classes=8)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img)
    img = img[np.newaxis, :]
     
    mask = predict_img(net=net,
                       full_img=img,
                       out_threshold=0.5,
                       device=device)
    
    mask = mask.transpose((1, 2, 3, 0))

    output = np.argmax(mask, axis=-1)
    output = output.astype(np.int16)
    print(output.shape)
    output = sitk.GetImageFromArray(output)
    # output = output.astype(np.int16)
    sitk.WriteImage(output, "data/sub015_predict_epoch15.nii.gz")