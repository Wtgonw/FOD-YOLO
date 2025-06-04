import cv2
import numpy as np

eps=np.finfo(np.double).eps

def simplest_color_balance(img_msrcr,s1,s2):
    sort_img=np.sort(img_msrcr,None)
    N=img_msrcr.size
    Vmin=sort_img[int(N*s1)]
    Vmax=sort_img[int(N*(1-s2))-1]
    img_msrcr[img_msrcr<Vmin]=Vmin
    img_msrcr[img_msrcr>Vmax]=Vmax
    return (img_msrcr-Vmin)*255/(Vmax-Vmin)

def gauss_blur(img,sigma,method='original'):
    if method=='original':
        return gauss_blur_original(img,sigma)
    elif method=='recursive':
        return gauss_blur_recursive(img,sigma)


def gauss_blur_original(img,sigma):
    row_filter=get_gauss_kernel(sigma,1)
    t=cv2.filter2D(img,-1,row_filter[...,None])
    return cv2.filter2D(t,-1,row_filter.reshape(1,-1))

def gauss_blur_recursive(img,sigma):
    pass

def get_gauss_kernel(sigma,dim=2):

    ksize=int(np.floor(sigma*6)/2)*2+1
    k_1D=np.arange(ksize)-ksize//2
    k_1D=np.exp(-k_1D**2/(2*sigma**2))
    k_1D=k_1D/np.sum(k_1D)
    if dim==1:
        return k_1D
    elif dim==2:
        return k_1D[:,None].dot(k_1D.reshape(1,-1))


def retinex_FM(img, iter=4):
    if len(img.shape) == 2:
        img = img[..., None]
    ret = np.zeros(img.shape, dtype='uint8')

    def update_OP(x, y):
        nonlocal OP
        IP = OP.copy()
        if x > 0 and y == 0:
            IP[:-x, :] = OP[x:, :] + R[:-x, :] - R[x:, :]
        if x == 0 and y > 0:
            IP[:, y:] = OP[:, :-y] + R[:, y:] - R[:, :-y]
        if x < 0 and y == 0:
            IP[-x:, :] = OP[:x, :] + R[-x:, :] - R[:x, :]
        if x == 0 and y < 0:
            IP[:, :y] = OP[:, -y:] + R[:, :y] - R[:, -y:]
        IP[IP > maximum] = maximum
        OP = (OP + IP) / 2

    for i in range(img.shape[-1]):
        R = np.log(img[..., i].astype('double') + 1)
        maximum = np.max(R)
        OP = maximum * np.ones(R.shape)
        S = 2 ** (int(np.log2(np.min(R.shape)) - 1))
        while abs(S) >= 1:
            for k in range(iter):
                update_OP(S, 0)
                update_OP(0, S)
            S = int(-S / 2)
        OP = np.exp(OP)
        mmin = np.min(OP)
        mmax = np.max(OP)
        ret[..., i] = (OP - mmin) / (mmax - mmin) * 255
    return ret.squeeze()


def MultiScaleRetinex(img, sigmas=[15, 80, 250], weights=None, flag=True):
    if weights == None:
        weights = np.ones(len(sigmas)) / len(sigmas)
    elif not abs(sum(weights) - 1) < 0.00001:
        raise ValueError('sum of weights must be 1!')
    r = np.zeros(img.shape, dtype='double')
    img = img.astype('double')
    for i, sigma in enumerate(sigmas):
        r += (np.log(img + 1) - np.log(gauss_blur(img, sigma) + 1)) * weights[i]
    if flag:
        mmin = np.min(r, axis=(0, 1), keepdims=True)
        mmax = np.max(r, axis=(0, 1), keepdims=True)
        r = (r - mmin) / (mmax - mmin) * 255
        r = r.astype('uint8')
    return r

def retinex_MSRCP(img, sigmas=[12, 80, 250], s1=0.01, s2=0.01):
    Int = np.sum(img, axis=2) / 3
    Diffs = []
    for sigma in sigmas:
        Diffs.append(np.log(Int + 1) - np.log(gauss_blur(Int, sigma) + 1))
    MSR = sum(Diffs) / 3
    Int1 = simplest_color_balance(MSR, s1, s2)
    B = np.max(img, axis=2)
    A = np.min(np.stack((255 / (B + eps), Int1 / (Int + eps)), axis=2), axis=-1)
    return (A[..., None] * img).astype('uint8')

def retinex_MSRCR(img, sigmas=[12, 80, 250], s1=0.01, s2=0.01):
    alpha = 125
    img = img.astype('double') + 1  #
    csum_log = np.log(np.sum(img, axis=2))
    msr = MultiScaleRetinex(img - 1, sigmas)  # -1
    r = (np.log(alpha * img) - csum_log[..., None]) * msr
    # beta=46;G=192;b=-30;r=G*(beta*r-b) #deprecated
    # mmin,mmax=np.min(r),np.max(r)
    # stretch=(r-mmin)/(mmax-mmin)*255 #linear stretch is unsatisfactory
    for i in range(r.shape[-1]):
        r[..., i] = simplest_color_balance(r[..., i], 0.01, 0.01)
    return r.astype('uint8')

def retinex_AMSR(img, sigmas=[12, 80, 250]):
    img = img.astype('double') + 1  #
    msr = MultiScaleRetinex(img - 1, sigmas, flag=False)
    # msr = retinex_MSRCR(img - 1, sigmas)
    y = 0.05
    for i in range(msr.shape[-1]):
        v, c = np.unique((msr[..., i] * 100).astype('int'), return_counts=True)
        sort_v_index = np.argsort(v)
        sort_v, sort_c = v[sort_v_index], c[sort_v_index]  # plot hist
        zero_ind = np.where(sort_v == 0)[0][0]
        zero_c = sort_c[zero_ind]
        #
        _ = np.where(sort_c[:zero_ind] <= zero_c * y)[0]
        if len(_) == 0:
            low_ind = 0
        else:
            low_ind = _[-1]
        _ = np.where(sort_c[zero_ind + 1:] <= zero_c * y)[0]
        if len(_) == 0:
            up_ind = len(sort_c) - 1
        else:
            up_ind = _[0] + zero_ind + 1
        #
        low_v, up_v = sort_v[[low_ind, up_ind]] / 100  # low clip value and up clip value
        msr[..., i] = np.maximum(np.minimum(msr[:, :, i], up_v), low_v)
        mmin = np.min(msr[..., i])
        mmax = np.max(msr[..., i])
        msr[..., i] = (msr[..., i] - mmin) / (mmax - mmin) * 255
    msr = msr.astype('uint8')
    return msr

