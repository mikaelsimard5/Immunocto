import torch.nn as nn
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from skimage import data, color


def random_uniform(r1, r2):
    return (r1 - r2) * torch.rand(3) + r2


def make_3d(x):
    return x.unsqueeze(-1).unsqueeze(-1)


class ColourAugment(nn.Module):
    """
    Colour augmentation based on:
    (1) A. C. Ruifrok and D. A. Johnston, “Quantification of histochemical staining by color deconvolution”.
    (2) the scikit-learn codes rgb2hed and hed2rgb (reimplemented here for torch tensors).
    (3) DOI: 10.1109/TMI.2018.2820199 for the perturbation scheme.
    """

    def __init__(self, sigma=0.05, mode='uniform'):
        super(ColourAugment, self).__init__()

        # In Ruifrok and Johnston's original paper, they do a 3-stain deconvolution with the last
        # one as DAB, with a stain vector [0.27, 0.57, 0.78]. In our case, we do H&E only. The
        # third vector can be calculated as the one orthogonal to the H and E vectors, calculable
        # with a cross product between the H and E stain vectors:
        H_stain_vector = [0.65, 0.70, 0.29]
        E_stain_vector = [0.07, 0.99, 0.11]
        residual = list(np.cross(np.array(H_stain_vector), np.array(E_stain_vector)))

        # Some references on the residual (although it's basic linear algebra)
        # https://blog.bham.ac.uk/intellimic/g-landini-software/colour-deconvolution-2/,
        # https://forum.image.sc/t/on-the-math-behind-colour-deconvolution-ruifrok-and-johnston-2001/66325

        self.rgb_from_hed = torch.tensor([H_stain_vector, E_stain_vector, residual], dtype=torch.float32)
        self.hed_from_rgb = torch.linalg.inv(self.rgb_from_hed)
        self.sigma = sigma
        self.mode = mode

    def rgb_to_stain(self, img, conv_matrix):

        c, h, w = img.shape
        img = img.reshape(img.shape[0], -1)  # collapse (C, H, W) to (C, H*W)
        torch.maximum(img, torch.tensor(1E-6), out=img)  # avoiding log artifacts
        log_adjust = torch.log(torch.tensor(1E-6))  # used to compensate the sum above
        stains = conv_matrix @ (torch.log(img) / log_adjust)

        return torch.maximum(stains, torch.tensor(0), out=stains).reshape(c, h, w)

    def stain_to_rgb(self, stains, conv_matrix):

        c, h, w = stains.shape
        stains = stains.reshape(stains.shape[0], -1)  # collapse (C, H, W) to (C, H*W)
        log_adjust = -torch.log(
            torch.tensor(1E-6))  # log_adjust here is used to compensate the sum within separate_stains().
        log_rgb = -conv_matrix @ (stains * log_adjust)
        rgb = torch.exp(log_rgb)
        return torch.clamp(rgb, min=0, max=1).reshape(c, h, w)

    def forward(self, img):  # rgb -> he
        # input img: float32 torch tensor(intensity ranging[0, 1]) of size (c, h, w)
        # output: colour-normalised float32 torch tensor of the same size and range, with colours perturbed.
        alpha, beta = torch.tensor(1.0), torch.tensor(0.0)

        conv_matrix_forward = torch.transpose(self.hed_from_rgb, 0, 1)
        stains = self.rgb_to_stain(img=img, conv_matrix=conv_matrix_forward)

        if self.mode == 'uniform':
            alpha = make_3d(random_uniform(r1=1 - self.sigma, r2=1 + self.sigma))
            beta = make_3d(random_uniform(r1=-self.sigma, r2=self.sigma))
        elif self.mode == 'normal':
            alpha = make_3d(torch.normal(mean=1.0, std=torch.tensor([self.sigma, self.sigma, self.sigma])))
            beta = make_3d(torch.normal(mean=0.0, std=torch.tensor([self.sigma, self.sigma, self.sigma])))

        stains_perturbed = alpha * stains + beta
        conv_matrix_backward = torch.transpose(self.rgb_from_hed, 0, 1)
        rgb_perturbed = self.stain_to_rgb(stains_perturbed, conv_matrix_backward)

        return rgb_perturbed

    def backward(self, stain):

        conv_matrix_backward = torch.transpose(self.rgb_from_hed, 0, 1)

        return self.stain_to_rgb(stains=stain, conv_matrix=conv_matrix_backward)


########################################################################################################################


if __name__ == '__main__':


    # Local image (H&E)
    #im0 = Image.open('/Users/mikael/Desktop/img.png').convert('RGB')
    #img = torch.permute(torch.tensor(np.asarray(im0) / 255.0, dtype=torch.float32), (2, 0, 1))

    # Cloud image (but not a good example as it's Hematoxylin & DAB (no Eosin)).
    ihc_rgb = data.immunohistochemistry() / 255.0
    img = torch.permute(torch.tensor(ihc_rgb, dtype=torch.float32), (2, 0, 1))

    # get stains
    m = ColourAugment(sigma=0.005, mode='uniform')
    c_f = torch.transpose(m.hed_from_rgb, 0, 1)
    c_b = torch.transpose(m.rgb_from_hed, 0, 1)
    ihc_hed = torch.permute(m.rgb_to_stain(img=img, conv_matrix=c_f), (1, 2, 0))
    null = torch.zeros_like(ihc_hed[:, :, 0])
    ihc_h = m.stain_to_rgb(stains=torch.permute(torch.stack((ihc_hed[:, :, 0], null, null), dim=-1), (2, 0, 1)),
                           conv_matrix=c_b)
    ihc_e = m.stain_to_rgb(stains=torch.permute(torch.stack((null, ihc_hed[:, :, 1], null), dim=-1), (2, 0, 1)),
                           conv_matrix=c_b)
    ihc_d = m.stain_to_rgb(stains=torch.permute(torch.stack((null, null, ihc_hed[:, :, 2]), dim=-1), (2, 0, 1)),
                           conv_matrix=c_b)

    # Display the decomposition of the target image
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img.numpy().transpose(1, 2, 0))
    ax[0].set_title("Original image")
    ax[1].imshow(ihc_h.numpy().transpose(1, 2, 0))
    ax[1].set_title("Hematoxylin")
    ax[2].imshow(ihc_e.numpy().transpose(1, 2, 0))
    ax[2].set_title("Eosin")  # Note that there is no Eosin stain in this image
    ax[3].imshow(ihc_d.numpy().transpose(1, 2, 0))
    ax[3].set_title("Residual")
    for a in ax.ravel():
        a.axis('off')
    fig.tight_layout()

    # Display colour augmentation examples of the target image
    fig, axes = plt.subplots(5, 5, figsize=(7, 7), sharex=True, sharey=True)
    AX = axes.ravel()
    for j in range(25):
        img_CA = torch.permute(m.forward(img=img), (1, 2, 0))
        AX[j].imshow(img_CA.numpy())
    for a in AX.ravel():
        a.axis('off')
    fig.tight_layout()