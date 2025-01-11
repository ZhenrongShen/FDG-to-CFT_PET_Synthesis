import os
import random
import logging
from datetime import datetime
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
import cv2
import SimpleITK as sitk
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .models.modules import weights_init
from .models.generator import Generator
from .models.discriminator import Discriminator
import monai


# ----------------------------------------
#             Reproducibility
# ----------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------
#                 Logging
# ----------------------------------------

# open the log file
def open_log(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if os.path.isfile(os.path.join(log_path, '{}.log'.format(log_name))):
        os.remove(os.path.join(log_path, '{}.log'.format(log_name)))
    initLogging(os.path.join(log_path, '{}.log'.format(log_name)))


# Init for logging
def initLogging(logFilename):
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s-%(levelname)s] %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S',
                        filename=logFilename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# ----------------------------------------
#          Data Type Conversion
# ----------------------------------------

def to_ndarray(tensor):
    """Converts a `torch.Tensor` to `numpy.ndarray`."""
    assert isinstance(tensor, torch.Tensor)
    return tensor.detach().cpu().numpy()


def to_tensor(array):
    """Converts a `numpy.ndarray` to `torch.Tensor`."""
    assert isinstance(array, np.ndarray)
    return torch.from_numpy(array).type(torch.FloatTensor).cuda()


# ----------------------------------------
#           Load & Save Networks
# ----------------------------------------

def load_model(process_net, pretrained_file):
    pretrained_dict = torch.load(pretrained_file)['model']
    process_net.load_state_dict(pretrained_dict)
    return process_net


def save_model(net, net_name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_name = os.path.join(save_path, net_name)
    net = net.module if isinstance(net, torch.nn.DataParallel) else net
    torch.save({'model': net.state_dict()}, model_name)


# ----------------------------------------
#             Create Networks
# ----------------------------------------

def create_generator(config, model_path=None):
    generator = Generator(
        ngf                 = config.GEN_FEAT_NUM,
        n_res_blocks        = config.NUM_RES_BLOCK,
        channel_multipliers = config.CHANNEL_MULTIPLIERS,
        use_spectral_norm   = config.SPECTRAL_NORM,
    )
    logging.info('Generator is created!')

    # Initialize the networks
    if model_path is not None:
        generator = load_model(generator, model_path)
        print(f'Load pre-trained generator from {model_path}')
    else:
        weights_init(generator, init_type=config.INIT_TYPE, init_gain=config.INIT_GAIN)
        logging.info('Initialize generator with %s type' % config.INIT_TYPE)

    return generator


def create_discriminator(config, model_path=None):
    discriminator = Discriminator(
        ndf               = config.DIS_FEAT_NUM,
        num_class         = config.CLASS_NUM,
        use_spectral_norm = config.SPECTRAL_NORM,
    )
    logging.info('Discriminator is created!')

    # Initialize the networks
    if model_path is not None:
        discriminator = load_model(discriminator, model_path)
        print(f'Load pre-trained discriminator from {model_path}')
    else:
        weights_init(discriminator, init_type=config.INIT_TYPE, init_gain=config.INIT_GAIN)
        logging.info('Initialize discriminator with %s type' % config.INIT_TYPE)

    return discriminator



def create_classifier(config, model_path=None):
    classifier = monai.networks.nets.resnet18(
        spatial_dims     = 3,
        n_input_channels = config.IN_CHANNELS,
        num_classes      = config.CLASS_NUM,
    )
    logging.info('Classifier is created!')

    # Initialize the networks
    if model_path is not None:
        classifier = load_model(classifier, model_path)
        logging.info(f'Load pre-trained Classifier from {model_path}')
    else:
        weights_init(classifier, init_type=config.INIT_TYPE, init_gain=config.INIT_GAIN)
        logging.info('Initialize Classifier with %s type' % config.INIT_TYPE)

    return classifier


# ----------------------------------------
#                 Metrics
# ----------------------------------------

def compute_psnr(image1, image2, denormalize=False, mask=None):
    # check image format
    assert isinstance(image1, torch.Tensor)
    assert isinstance(image2, torch.Tensor)
    assert image1.shape == image2.shape, "Input images must have the same shape"

    # transform data type
    if denormalize:
        image1 = torch.clamp((image1.clone() + 1.0) / 2, min=0.0, max=1.0)
        image2 = torch.clamp((image2.clone() + 1.0) / 2, min=0.0, max=1.0)
    else:
        image1 = torch.clamp(image1.clone(), min=0.0, max=1.0)
        image2 = torch.clamp(image2.clone(), min=0.0, max=1.0)
    image1 = to_ndarray(torch.squeeze(image1))
    image2 = to_ndarray(torch.squeeze(image2))

    # compute metrics
    if mask is not None:
        assert isinstance(mask, np.ndarray)
        mask_bool = mask.astype(bool)
        image_pred = image1[mask_bool]
        image_true = image2[mask_bool]
    else:
        image_pred = image1
        image_true = image2
    psnr_value = peak_signal_noise_ratio(image_true, image_pred, data_range=1.0)
    return psnr_value


def compute_ssim(image1, image2, denormalize=False, mask=None):
    # check image format
    assert isinstance(image1, torch.Tensor)
    assert isinstance(image2, torch.Tensor)
    assert image1.shape == image2.shape, "Input images must have the same shape"

    # transform data type
    if denormalize:
        image1 = torch.clamp((image1.clone() + 1.0) / 2, min=0.0, max=1.0)
        image2 = torch.clamp((image2.clone() + 1.0) / 2, min=0.0, max=1.0)
    else:
        image1 = torch.clamp(image1.clone(), min=0.0, max=1.0)
        image2 = torch.clamp(image2.clone(), min=0.0, max=1.0)
    image1 = to_ndarray(torch.squeeze(image1))
    image2 = to_ndarray(torch.squeeze(image2))

    # compute metrics
    if mask is not None:
        assert isinstance(mask, np.ndarray)
        mask_bool = mask.astype(bool)
        image_pred = image1[mask_bool]
        image_true = image2[mask_bool]
    else:
        image_pred = image1
        image_true = image2
    ssim_value = structural_similarity(image_true, image_pred, data_range=1.0)
    return ssim_value


def get_SUVR_image(image, ref_mask, denormalize=False):
    # check and transform image format
    if isinstance(image, torch.Tensor):
        if denormalize:
            image = torch.clamp((image.clone() + 1.0) / 2, min=0.0, max=1.0)
        else:
            image = torch.clamp(image.clone(), min=0.0, max=1.0)
        image = to_ndarray(torch.squeeze(image))

    # compute SUVR values
    ref_mask_bool = ref_mask.astype(bool)
    masked_values = image[ref_mask_bool]
    mean_value = np.mean(masked_values)
    suvr_image = image / mean_value
    return suvr_image


# DAT biomarker - striatal binding ratio (SBR) 
def compute_SBR(syn_image, ref_mask, denormalize=False, target_mask=None):
    syn_image_suvr = get_SUVR_image(syn_image, ref_mask, denormalize)

    suvr_mean_ref = np.mean(syn_image_suvr[ref_mask.astype(bool)])
    if target_mask is not None:
        suvr_mean_target = np.mean(syn_image_suvr[target_mask.astype(bool)])
    else:
        suvr_mean_target = np.mean(syn_image_suvr)

    sbr = (suvr_mean_target - suvr_mean_ref) / suvr_mean_ref
    return sbr


def find_youden_index_threshold(fpr, tpr, thresholds):
    """ Youden's index = sensitivity + specificiy - 1 = sensitivity - (1 - specificiy) = TPR - FPR."""
    diff = tpr - fpr
    max_index = diff.argmax()
    return thresholds[max_index]


# ----------------------------------------
#               Save Results
# ----------------------------------------

def get_error_map(pred_images, gt_images):
    error_map_list = []
    for i in range(pred_images.shape[0]):
        error_map = gt_images[i] - pred_images[i]
        error_map_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min())  # [0, 1]
        error_map_norm = torch.clamp(error_map_norm * 2 - 1, min=-1, max=1)                   # [-1, 1]
        error_map_list.append(error_map_norm)
    return torch.stack(error_map_list).cuda()


def show_image(imgs, img_name, save_path, denormalize=True, grid_nrow=8):
    # Create save path
    save_path = os.path.join(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, img_name)

    # Denormalization [-1, 1] --> [0, 1]
    if isinstance(imgs, list):
        grid_nrow = imgs[0].shape[0]
        imgs = torch.cat(imgs, dim=0)

    if denormalize:
        out_imgs = torch.clamp((imgs + 1) / 2, min=0.0, max=1.0)
    else:
        out_imgs = torch.clamp(imgs, min=0.0, max=1.0)

    # Save images
    grid = make_grid(out_imgs, nrow=grid_nrow)
    save_image(grid, filename)

    # Apply color map
    im_data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    color_image = cv2.applyColorMap(im_data, cv2.COLORMAP_HOT)
    cv2.imwrite(filename, color_image)


def save_vol_image(vol_img, img_name, save_path, denormalize=True, spacing=2.0):
    # Create save path
    save_path = os.path.join(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, img_name)

    # Denormalization [-1, 1] --> [0, 1]
    if denormalize:
        vol_img = torch.clamp((vol_img.clone() + 1) / 2, min=0.0, max=1.0)
    else:
        vol_img = torch.clamp(vol_img.clone(), min=0.0, max=1.0)

    # Tensor --> SimpleITK Image
    vol_img_np = vol_img.squeeze().detach().cpu().numpy()
    out_img = sitk.GetImageFromArray(vol_img_np[:, ::-1, :])

    # Set spacing
    spacings = (spacing, spacing, spacing)
    out_img.SetSpacing(spacings)

    # Save image
    sitk.WriteImage(out_img, filename)
