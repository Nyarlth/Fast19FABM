import numpy as np 
from zernike import RZern as rz
from numpy.linalg import lstsq, matrix_rank, norm
from matplotlib import pyplot as plt 
from scipy import  interpolate
Rbf = interpolate.Rbf
import gc
from multiprocessing import Pool
import time
class RZern(rz):
    def fit_cart_grid(self, Phi, rcond=None, apply_zn=[], mask = None):
        """Fit a cartesian grid using least-squares.

        Parameters
        ----------
        - `Phi`: cartesian grid, e.g., generated with make_cart_grid().
        - `rcond`: rcond supplied to `lstsq`

        Returns
        -------
        -   `a`, `numpy` vector of Zernike coefficients
        -   `res`, see `lstsq`
        -   `rnk`, see `lstsq`
        -   `sv`, see `lstsq`

        Examples
        --------

        .. code:: python

            import numpy as np
            import matplotlib.pyplot as plt
            from zernike import RZern

            cart = RZern(6)
            L, K = 200, 250
            ddx = np.linspace(-1.0, 1.0, K)
            ddy = np.linspace(-1.0, 1.0, L)
            xv, yv = np.meshgrid(ddx, ddy)
            cart.make_cart_grid(xv, yv)

            c0 = np.random.normal(size=cart.nk)
            Phi = cart.eval_grid(c0, matrix=True)
            c1 = cart.fit_cart_grid(Phi)[0]
            plt.figure(1)
            plt.subplot(1, 2, 1)
            plt.imshow(Phi, origin='lower', extent=(-1, 1, -1, 1))
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.plot(range(1, cart.nk + 1), c0, marker='.')
            plt.plot(range(1, cart.nk + 1), c1, marker='.')

            plt.show()

        """
        vPhi = self.vect(Phi)
        zfm = np.logical_and(np.isfinite(self.ZZ[:, 0]), np.isfinite(vPhi))
        zfm = np.logical_and(zfm, vPhi != 0)
        
        try:
            zfm = np.logical_and(zfm, ~mask.ravel(order='F'))
        except:
#             print("mask data error!!!")
            pass
        if apply_zn:
            zn = np.zeros(self.nk)
        else:
            zn = np.ones(self.nk)
        for i in apply_zn:
            zn[i] = 1
        #print(self.ZZ.shape)
        zfA = self.ZZ[zfm,:][:,zn==1]
        Phi1 = vPhi[zfm]

        a, res, rnk, sv = lstsq(np.dot(zfA.T, zfA),
                                np.dot(zfA.T, Phi1),
                                rcond=rcond)
        zn[zn==1] = a
        return zn, res, rnk, sv

def Zfitting(fig, fig_size = None , order = 19, apply_mod = [], apply_cut = None, mask = None):
    '''
    参数
    ---
    fig: 输入的图像，必须是二维的
    fig_size: 输入图像的大小，函数只拟合原点半径1以内的区域
    oder: 用于拟合的zernike多项式的阶数
    apply_mod: 用于拟合的多项式的序号，默认全部选用

    返回值
    ---
    Phi: 拟合后的图像
    c1:  各个zernike多项式的权重
    '''
    cart = RZern(order)
    L, K = fig.shape
    if fig_size:
        try:
            ddx = np.linspace(-fig_size[1], fig_size[1], K)
            ddy = np.linspace(-fig_size[0], fig_size[0], L)
        except:
            ddx = np.linspace(-fig_size, fig_size, K)
            ddy = np.linspace(-fig_size, fig_size, L)
    else:
        ddx = np.linspace(-1.0 + 1/K, 1.0 - 1/K, K)
        ddy = np.linspace(-1.0 + 1/L, 1.0 - 1/L, L)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    c1  = cart.fit_cart_grid(fig, apply_zn = apply_mod, mask = mask)[0]
    if apply_cut:
        c1_ = abs(c1)
        mx = c1_.max()
        c1[c1_ < mx * apply_cut] = 0
    Phi = cart.eval_grid(c1, matrix = True)
    return Phi, c1

if __name__ == "__main__":
    a = np.ones((100, 550))
    b = Zfitting(a)
    print(b[0].shape, b[1].shape)

class Beam_cart():
    def __init__(self, Z_order = 23, size = 10):
        '''
        size: the beam fitting size, the radius [arcmin]
        Z_order: the max Zernike fitting order
        '''
        self.cart = RZern(Z_order)
        self.size = size
    def slice_data(self, coef, dist_, angl):
        cart = self.cart
        dist = dist_ / self.size
        try:
            ans = np.zeros((cart.nk,) +  dist.shape)
        except:
            ans = np.zeros((len(index), len(dist)))
        mask = dist>1
        for i in range(cart.nk):
            k = i
            if coef[k]:
                ans[i] = cart.radial(k, dist) * cart.angular(k, angl) * coef[k]
                
        ans[:,mask] = 0
        gc.collect()
        return ans.sum(axis = 0)

def order(m,n):
    # index the ZP mode $Z_{n}^{m}$ in the Wyant index
    if (n+m)%2:
        print("error, no such order")
        raise
    if abs(m)>n:
        print("error, no such order")
        raise
    return RZern.nm2noll(n,m)-1
#     if n:
#         return (n+1)*n//2-1*(m>0)+abs(m)
#     else:
#         return 0

def mapping(source, sky, coef, cart = Beam_cart(), rot = 0):
    '''
    source: list of sources ra.dec[ra,dec,flux], ra,dec need in ndarray type
    sky:    list of sky map ra.dec[ra,dec], ra,dec need in ndarray type, need reshape in 1D dim
    return: shape of (len(source.ra), len(sky.ra))
    '''
    _ra, _dec, _flux = source
    _ra  = _ra.astype('float64')
    _dec = _dec.astype('float64')
    sky_ra, sky_dec = sky
    sky_ra  = sky_ra.astype('float64')
    sky_dec = sky_dec.astype('float64')
    r = np.sin(np.radians(_dec[:,None])) * np.sin(np.radians(sky_dec[None,:])) \
      + np.cos(np.radians(_dec[:,None])) * np.cos(np.radians(sky_dec[None,:])) \
      * np.cos(np.radians(_ra[:,None]) - np.radians(sky_ra[None,:]))
    r[r>1] = 1.
    r = np.arccos(r)

    # parallel angle of the source
    p = (np.sin(r) * np.cos(np.radians(_dec[:,None])))
    p[p==0] = np.inf
    p = (np.sin(np.radians(sky_dec[None,:]))-np.cos(r)*np.sin(np.radians(_dec[:,None])))/p
    p[p>1] = 1.
    p = np.arccos(p) * np.sign(_ra[:,None] - sky_ra[None,:])
    r *= 180./np.pi * 60. # arcmin
    p *= 180./np.pi
    theta = -90 - p  + rot
#     os.system("echo 1 >>ttlog")
    gc.collect()
#     print(f"Sources RA range {_ra[0]} -- {_ra[-1]} start at {time.ctime()}")
    t = (cart.slice_data(coef, r, theta / 180 * np.pi) * _flux[:,None]).sum(axis=0)
#     print(f"Sources RA range {_ra[0]} -- {_ra[-1]} end   at {time.ctime()}")
    return t

def mp_map(x): 
    return mapping(*x[:3], cart = x[3], rot = x[4])

def sky_map(cata, coef, sky_range = None, Z_order=23, rot = 0, Beam_size = 10, delta = 1, processes = 5, Batch_s = 200):
    '''
    cata [3+, n], (ra, dec, flux,...) * n
    '''
    ra   = cata[0]
    dec  = cata[1]
    flux = cata[2]
    if not sky_range:
        sky_range = [ra[0] - Beam_size/60, ra[-1] + Beam_size/60, dec.min() - Beam_size/60, dec.max() +Beam_size/60]
    ra_edges  = np.arange(*sky_range[:2], step = delta/60)
    dec_edges = np.arange(*sky_range[2:], step = delta/60)
    
    RA, DEC = np.meshgrid(ra_edges, dec_edges)
    Ndec = len(dec_edges)
    Nsource = len(ra)
    sky = np.zeros_like(RA)
    
    cart = Beam_cart(Z_order=Z_order, size = Beam_size)
    args = []
    sky_temp = []
    for i in range(Nsource//Batch_s +1):
        sl    = slice(i*Batch_s, i*Batch_s+Batch_s)
        ra_   = ra[sl]
        dec_  = dec[sl]
        flux_ = flux[sl]
        sl,sr = (int(max(ra_.min() - Beam_size/60 - sky_range[0],0) // (delta / 60)), int((ra_.max() + Beam_size/60 - sky_range[0]) // (delta / 60)))
#         print(sl, sr)
        args.append(((ra_,dec_,flux_), (RA[:,sl:sr].reshape(-1), DEC[:,sl:sr].reshape(-1)), coef, cart, 23.4))
        sky_temp.append(sky[:,sl:sr])
    with Pool(processes = processes) as p:
        results = p.map(mp_map,args)
    
    for i in range(len(sky_temp)):
        sky_temp[i] += results[i].reshape(Ndec,-1)
    return sky, ra_edges, dec_edges
    
#     return RA, DEC
    




















def plot_coef(ax, coef, Z_order):
    '''
    coef the list of coeffient energy
    Z_order the max zernike n value
    '''
    coef = np.array(coef)
    q_0 = []
    for i in range(0, Z_order, 2):
        q_0.append(order(0,i))
    q_1 = []
    for i in range(1, Z_order, 2):
        q_1.append(order(-1,i))
        q_1.append(order( 1,i))
        
    q_2 = []
    for i in range(2, Z_order, 2):
        q_2.append(order(-2,i))
        q_2.append(order( 2,i))
        
    q_ = []
    qt = q_0 + q_1 + q_2
    for i in range(len(coef)):
        if i in qt:
            pass
        else:
            q_.append(i)
    ax.axhline(max(coef[q_0]) * 0.01, linestyle = '--', lw = 2, c = 'k')
    ax.scatter(q_0, coef[q_0], c = 'r'        , marker = 'o', edgecolors='r',label =r"$Z_n^0$")
    ax.scatter(q_0, -coef[q_0], c = (0,0,0,0) , marker = 'o', edgecolors='r')
    
    
    ax.scatter(q_1, coef[q_1], c = "g"        , marker = 'o', edgecolors='g',label =r"$Z_n^{\pm 1}$")
    ax.scatter(q_1, -coef[q_1], c = (0,0,0,0) , marker = 'o', edgecolors='g')
    
    
    ax.scatter(q_2, coef[q_2], c = "b"       , marker = 'o', edgecolors='b',label =r"$Z_n^{\pm 2}$")
    ax.scatter(q_2, -coef[q_2], c = (0,0,0,0) , marker = 'o', edgecolors='b')
    
    
    ax.scatter(q_, coef[q_], c = "y"       , marker = 'o', edgecolors='y',label =r"$Z_n^{\ge 3}$")
    ax.scatter(q_, -coef[q_], c = (0,0,0,0) , marker = 'o', edgecolors='y')
    ax.scatter(0,1e-5, c = "k", marker = "o", edgecolors="k", label = "Positive")
    ax.scatter(0,1e-5, c = (0,0,0,0), marker = "o", edgecolors="k", label = "Negative")
    
    ax.set_ylim([1e-4,0.15])
    
    
    return ax

def plot_single_coef(coef, Z_order, filename = None):
    '''
    coef the list of coeffient energy
    z_order the max zernike n value
    '''
    ax = plot_coef(plt.gca(), coef, Z_order)
    ax.set_yscale("log")
    plt.legend()
    ax.set_xticks(list(range(0, len(coef), 50)))
    ax.set_xlabel("Noll indices")
    ax.set_ylabel("Fitting coeffient")
    if filename:
        plt.savefig(filename + ".png", format = "png", dpi = 100)
