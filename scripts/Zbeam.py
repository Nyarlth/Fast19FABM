import numpy as np 
from zernike import RZern 
import gc


class Beam_cart():
    def __init__(self, Z_order = 23, size = 10):
        '''
        size: the beam fitting size, the radius [arcmin]
        Z_order: the max Zernike fitting order
        '''
        self.cart = RZern(Z_order)
        self.size = size
        
    def make_grid(self, H_coor,V_coor):
        '''
        Input the horizontal and vertical edges [arcmin]
        '''
        K,L = len(H_coor)-1, len(V_coor)-1
        ddx = np.linspace(H_coor[0]/self.size + 1/K, H_coor[-1]/self.size - 1/K, K)
        ddy = np.linspace(V_coor[0]/self.size + 1/L, V_coor[-1]/self.size - 1/L, L)
        xv, yv = np.meshgrid(ddx, ddy)
        self.cart.make_cart_grid(xv, yv)
    
    def order(m,n):
        '''
        ZP mode to the list Index
        '''
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

    
    def eval_grid(self, coef):
        '''
        Cal the rresponse value of the input grid.
        '''
        return self.cart.eval_grid(coef, matrix = True) 
    
    def slice_data(self, coef, dist_, angl):
        '''
        Output the response in polar coordinate.
        The dist_ is the radius list in unit [arcmin].
        '''
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
    
    def mapping(self, source, sky, coef, rot = 0):
        '''
        source: list of sources ra.dec[RA., decl., flux]. RA., decl., flux need in ndarray type
        sky:    list of sky map ra.dec[RA., decl.]. RA., decl. need in ndarray type, need reshape in 1D dim
        return: shape of (len(sky.RA), )
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
        t = (self.slice_data(coef, r, theta / 180 * np.pi) * _flux[:,None]).sum(axis=0)
    #     print(f"Sources RA range {_ra[0]} -- {_ra[-1]} end   at {time.ctime()}")
        return t