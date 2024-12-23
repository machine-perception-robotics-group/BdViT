import os
import scipy.io
import tarfile

if __name__ == '__main__':
    imagenet_valid_tar_path = './ILSVRC2012_img_val.tar'
    target_dir = './ILSVRC2012_img_val_for_ImageFolder'
    meta_path = './ILSVRC2012_devkit_t12/data/meta.mat'
    truth_label_path = './ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'

    meta = scipy.io.loadmat(meta_path, squeeze_me=True)
    ilsvrc2012_id_to_wnid = {m[0]: m[1] for m in meta['synsets']}

    with open(truth_label_path, 'r') as f:
        ilsvrc_ids = tuple(int(ilsvrc_id) for ilsvrc_id in f.read().split('\n')[:-1])

    for ilsvrc_id in ilsvrc_ids:
        wnid = ilsvrc2012_id_to_wnid[ilsvrc_id]
        os.makedirs(os.path.join(target_dir, wnid), exist_ok=True)

    os.makedirs(target_dir, exist_ok=True)
    num_valid_images = 50000
    with tarfile.open(imagenet_valid_tar_path, mode='r') as tar:
        for valid_id, ilsvrc_id in zip(range(1, num_valid_images+1), ilsvrc_ids):
            wnid = ilsvrc2012_id_to_wnid[ilsvrc_id]
            filename = 'ILSVRC2012_val_{}.JPEG'.format(str(valid_id).zfill(8))
            print(filename, wnid)
            img = tar.extractfile(filename)
            with open(os.path.join(target_dir, wnid, filename), 'wb') as f:
                f.write(img.read())
