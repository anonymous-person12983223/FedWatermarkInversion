from torchvision import transforms

def trigger_image(image, pattern, channels=3, batch=False):
    """
    Apply a backdoor trigger pattern to an image or batch of images using alpha blending.

    Args:
        image (Tensor): The original image tensor. Shape:
            - (C, H, W) if batch=False
            - (B, C, H, W) if batch=True
        pattern (Tensor): The trigger pattern tensor with alpha channel.
            - Shape (H, W, C+1) for single image
            - Shape (B, H, W, C+1) after unsqueeze for batch
        channels (int): Number of image channels (e.g., 3 for RGB).
        batch (bool): If True, processes a batch of images.

    Returns:
        Tensor: Image(s) with the trigger pattern applied.
    """
    if not batch:
        # Split alpha and RGB(A) channels
        alpha_channel = pattern[..., channels:]
        # Alpha blend pattern and image: alpha * pattern + (1 - alpha) * original
        composite = alpha_channel * pattern[..., :channels] + (1 - alpha_channel) * image
        # Reorder from HWC to CHW
        composite = composite.permute(2, 0, 1)
    else:
        # Expand pattern for batch and reorder to NCHW
        pattern_batch = pattern.unsqueeze(0)
        pattern_batch = pattern_batch.permute(0, 3, 1, 2)
        alpha_channel = pattern_batch[:, channels:, :, :]
        composite = alpha_channel * pattern_batch[:, :channels, :, :] + (1 - alpha_channel) * image
    return composite


def normalize(image, channels=3):
    """
    Normalize an image tensor using mean and std of 0.5 for each channel.

    Args:
        image (Tensor): Input image tensor of shape (C, H, W).
        channels (int): Number of channels to normalize (default is 3).

    Returns:
        Tensor: Normalized image tensor.
    """
    norm = transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
    return norm(image)
