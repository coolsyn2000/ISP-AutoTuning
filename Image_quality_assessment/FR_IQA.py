import lpips


def loss_lpips(test_img_path, ref_img_path, net='alex'):
    #spatial = True  # Return a spatial map of perceptual distance.

    loss_fn = lpips.LPIPS(net=net)
    test_img = lpips.im2tensor(lpips.load_image(test_img_path))
    ref_img = lpips.im2tensor(lpips.load_image(ref_img_path))
    dist = loss_fn.forward(ref_img, test_img)

    return dist.item()
