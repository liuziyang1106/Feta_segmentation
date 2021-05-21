
import numpy as np
import torch,os,shutil
import torch.nn.functional as F
from model_zoo import UNet
import nibabel as nib


def predict_mask(net, full_img, device, out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(full_img)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        full_mask = probs.squeeze().cpu().numpy()
    mask = np.argmax(full_mask > out_threshold, axis=0).astype(np.int16)
    return mask

def Inference_single_image(net, model_ckpt, img_path, output_path='./'):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
    net.to(device=device)
    net.load_state_dict(torch.load(model_ckpt, map_location=device)['state_dict'])

    img = nib.load(img_path)
    img_data = img.get_fdata()
    img_affine = img.affine
    img_data = img_data[np.newaxis, :]
    sub_id = (img_path.split('/')[-1]).replace('T2w', 'pred')
    print(sub_id)
    mask = predict_mask(net=net,full_img=img_data,out_threshold=0.5,device=device)
    inference_path = os.path.join(output_path, sub_id)
    predict_img = nib.Nifti1Image(mask, img_affine).to_filename(inference_path)

def Inference_Folder_images(net, model_ckpt, folder_path, output_path='./'):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
    net.to(device=device)
    net.load_state_dict(torch.load(model_ckpt, map_location=device)['state_dict'])

    for img_path in os.listdir(folder_path):
        if 'T2w' in img_path:
            img = nib.load(os.path.join(folder_path,img_path))
            img_data = img.get_fdata()
            img_affine = img.affine
            img_data = img_data[np.newaxis, :]
            sub_id = (img_path.split('/')[-1]).replace('T2w', 'pred')
            print(sub_id)
            mask = predict_mask(net=net,full_img=img_data,out_threshold=0.5,device=device)
            inference_path = os.path.join(output_path, sub_id)
            predict_img = nib.Nifti1Image(mask, img_affine).to_filename(inference_path)
            shutil.copyfile(os.path.join(folder_path,img_path),os.path.join(output_path, img_path))


if __name__ == "__main__":
    model = "./runs/test_64*128*128/unet_best_model.pth.tar"
    img_path = "data/imgs_crop/sub-015_T2w.nii.gz"
    folder = "data/test/"
    net = UNet(n_channels=1, n_classes=8)
    # Inference_single_image(net, model, img_path)
    Inference_Folder_images(net, model, folder,'data')

    

