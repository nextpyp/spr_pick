from __future__ import print_function,division

import numpy as np

#from topaz.utils.picks import as_mask
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # h[h > 0.4] = 1
    # h[h < 0.1] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def as_mask(shape, x_coord, y_coord, radii):

    ygrid = np.arange(shape[0])
    xgrid = np.arange(shape[1])
    xgrid,ygrid = np.meshgrid(xgrid, ygrid, indexing='xy')
    mask = np.zeros(shape, dtype=np.uint8)
    for i in range(len(x_coord)):
        x = x_coord[i]
        y = y_coord[i]
        radius = radii[i]
        threshold = radius**2
        
        d2 = (xgrid - x)**2 + (ygrid - y)**2
        mask += (d2 <= threshold)

    mask = np.clip(mask, 0, 1)
    return mask

def as_gaussian(shape, x_coord, y_coord, bb = 36):
    hm = np.zeros(shape, dtype=np.float32)-1
    draw_gaussian = draw_umich_gaussian
    radius = gaussian_radius((bb, bb))
    radius = max(0, int(radius))
    # print('radius', radius)
    for i in range(len(x_coord)):
        x = x_coord[i]
        y = y_coord[i]
        ct_int = np.array([x,y]).astype(np.int32)
        draw_gaussian(hm, ct_int, radius)
    return hm

def coordinates_table_to_dict(coords):
    root = {}
    if 'source' in coords:
        for (source,name),df in coords.groupby(['source', 'image_name']):
            xy = df[['x_coord','y_coord']].values.astype(np.int32)
            root.setdefault(source,{})[name] = xy
    else:
        for name,df in coords.groupby('image_name'):
            xy = df[['x_coord','y_coord']].values.astype(np.int32)
            root[name] = xy
    return root

def match_coordinates_to_images(coords, images, gt_images = None, radius=-1, bb=32):
    """
    If radius >= 0, then convert the coordinates to an image mask
    """
    
    nested = 'source' in coords
    coords = coordinates_table_to_dict(coords)
    null_coords = np.zeros((0,2), dtype=np.int32)
    # print('bb,', bb)
    matched = {}
    if nested:
        # print('nested ')
        for source in images.keys():
            this_matched = matched.setdefault(source,{})
            this_images = images[source]
            if gt_images is not None:
                this_gt = gt_images[source]
            else:
                this_gt = None 
            this_coords = coords.get(source, {})
            for name in this_images.keys():
                im = this_images[name]
                if this_gt is not None:
                    gt = this_gt[name]
                xy = this_coords.get(name, null_coords)

                if radius >= 0:
                    radii = np.array([radius]*len(xy), dtype=np.int32)
                    shape = (im.height, im.width)
                    shape_small = (int(im.height//2), int(im.width//2))
                    xys = as_mask(shape, xy[:,0], xy[:,1], radii)
                    print('bb', bb)
                    hm = as_gaussian(shape, xy[:,0], xy[:,1], bb=bb)
                    hm_small = as_gaussian(shape_small,np.ceil(xy[:,0]//2), np.ceil(xy[:,1]//2), bb=bb//2)
                if this_gt is not None:
                    this_matched[name] = (im, gt, xys, hm, hm_small)
                else:
                    this_matched[name] = (im,xys, hm, hm_small)
    else:
        for name in images.keys():
            im = images[name]
            if gt_images is not None:
                gt = gt_images[name]
            xy = coords.get(name, null_coords)
            if radius >= 0:
                radii = np.array([radius]*len(xy), dtype=np.int32)
                shape = (im.height, im.width)
                shape_small = (int(im.height//2), int(im.width//2))
                xys = as_mask(shape, xy[:,0], xy[:,1], radii)
                hm = as_gaussian(shape, xy[:,0], xy[:,1], bb=bb)
                hm_small = as_gaussian(shape_small,np.ceil(xy[:,0]//2), np.ceil(xy[:,1]//2), bb=bb//2)
            if gt_images is not None:
                matched[name] = (im,gt,xys, hm, hm_small)
            else:
                matched[name] = (im, xys, hm, hm_small)
    return matched 
    





