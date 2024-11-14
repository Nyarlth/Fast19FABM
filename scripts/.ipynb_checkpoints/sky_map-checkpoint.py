from Zfitting import *
import h5py as h5

with h5.File("../data/19FA.h5",'r') as f:
    print(f.keys())
    print(f.attrs.keys())
    B = f["Origin"][:]
    Zf = f["Fitted"][:]
    coef = f["Coefficient"][:]
    Z_order = f.attrs["Z_order"]
    xx = f.attrs["X_coor"][:]
    yy = f.attrs["Y_coor"][:]
    fig_size = f.attrs["fig_size"]
    polar = f.attrs["polar"]

    
cata = np.load("../../beamstacking/script/FATH_FULL_CATA.npy",allow_pickle=True)
a = cata.T
ind = np.lexsort(a.T[::-1])
cata_s = a[ind]
cata_s = cata_s.T
sky, ra_e, dec_e = sky_map(cata_s, coef[8], processes = 20)

with h5.File("../data/sky_map_.h5","w") as f:
    f["sky"] = sky 
    f["ra_edge"] = ra_e
    f["dec_edge"] = dec_e
print("Done!")