# {1}
FOLDER = "D:\Stack\Process"
THRESHOLD_MULTIPLIER = 3.5
MAGNITUDE_SHIFT = 7.23-1.5+0.243
CATALOG_MAGNITUDE_LIMIT = 4.5
COLOR_INDEX_DEVIATION_LIMIT = 0.75
FINDING_STARS_MAGNITUDE_DIVERGENCE_LIMIT = 100
FINDING_STARS_RANGE_LIMIT = 25
from math import pi
LATITUDE = (57+2/60+12/3600)/180*pi
ASTRAL_TIME = (21+26/60+1/3600)/12*pi
AVERAGING_AMOUNT = 1
DEBUG = False
COLOR_INDEX_ANOTATE = False

# {2}
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
from math import acos, cos, sqrt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats 
from photutils.detection import DAOStarFinder
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import os
from sklearn.linear_model import LinearRegression
from os.path import exists
from ctapipe.utils import get_bright_stars
path = os.getcwd()

# {3}
def get_pixs(coords, hdu, index):
        wcs = WCS(hdu[0].header)
        shape = hdu[0].shape
        for coord in coords:
            mass = [float(s) for s in wcs.world_to_pixel(SkyCoord(*coord[index],unit="deg"))]
            if 0 <= mass[0] < shape[0] and 0 <= mass[1] < shape[1]: coord.append(mass)

# {4}
def download():
    # gets star database 
    """(get_bright_stars works only on Linux so change "HOME" to "USERPROFILE" when KeyError occurs)"""
    if not exists("values.dat"):
        data = get_bright_stars()
        data.write('values.dat', format='ascii')  

# {4}
def sort(hdu, target_color_index=0.0,deviation=0.2, max_magnitude = 5):
    w = WCS(hdu[0].header)
    shape = hdu[0].shape
    ra,dec = [float(s) for s in w.pixel_to_world(shape[0]/2,shape[1]/2).to_string().split()]
    rae,dece = [float(s) for s in w.pixel_to_world(0,0).to_string().split()]
    rah,dech = [float(s) for s in w.pixel_to_world(shape[0]-1,shape[1]-1).to_string().split()]
    r = max(dist(*dtr(ra,dec,rae,dece)),dist(*dtr(ra,dec,rah,dech)))
    # filtrates and sorts database in values.dat file
    lst = []
    with open(path+"\\values.dat") as f:
        f.readline()
        for ln in f:
            out = ""
            flag = True
            for i in ln:
                if i == "\"": flag = not flag
                out += i if flag else ""
            _, nm, __, Vmag, BV, ___, ra_dec = out.split()
            if Vmag == "\"" or BV == "\"": continue 
            ra_dec = ra_dec.split(',')
            if float(Vmag) <= max_magnitude:
                Vmag, BV, RA, DEC = [float(s) 
                        for s in [Vmag,BV,ra_dec[0],ra_dec[1]]]
                if dist(*dtr(ra,dec,RA,DEC)) < r and abs(BV - target_color_index) <= deviation:
                    lst.append([Vmag, BV, (RA, DEC)])
    return lst

# {5}
def hav(a): return (1-cos(a))/2
def ahav(a): return acos(1-2*a)
def dist(ra1,dec1,ra2,dec2): 
    return ahav(hav(abs(dec1-dec2))+cos(dec1)*cos(dec2)*hav(abs(ra1-ra2)))

# {6}
def dtr(*args): # deg to rad
    return [i/180*pi for i in args]

# {7}
def star_finder(hdu, median = None, std = None):
    data = hdu[0].data
    if (std == None) != (median == None): raise ValueError
    if std == None: mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=std*THRESHOLD_MULTIPLIER)  
    sources = daofind(data - median) 
    measured_data = sorted(np.transpose((sources['xcentroid'], sources['ycentroid'],sources['mag'])).tolist())
    for val in measured_data: val[2] += MAGNITUDE_SHIFT
    return measured_data, median, std

# {8}
def match(real_data,measured_data):
    comp = []
    for rd in real_data:
        if len(rd)==4:
            r = FINDING_STARS_RANGE_LIMIT**2
            rm,BV,ra_dec,(rx,ry) = rd
            o = 0
            for mx,my,mm in measured_data:
                if abs(mm-rm)<FINDING_STARS_MAGNITUDE_DIVERGENCE_LIMIT:
                    if (rx-mx)**2+(ry-my)**2<r:
                        r = (rx-mx)**2+(ry-my)**2
                        fmm = mm; fmx = mx; fmy = my
                        o = 1
            if o:
                comp.append([rm, fmm, BV, ra_dec, air_masses(*ra_dec)])
                if DEBUG:
                    plt.plot(rx,ry,'x',color='green')
                    plt.plot(fmx,fmy,'x', color='red')
                    plt.annotate(str(int(acos(1/comp[-1][-1])/pi*180)), (rx,ry), color="white")
            elif DEBUG: 
                print(f"Failed to find mathing star: ({rx}, {ry})")
    return comp

# {9}
def air_masses(ra1,dec1):
    ra1, dec1 = ra1*pi/180, dec1*pi/180
    return 1/cos(dist(ra1,dec1,ASTRAL_TIME,LATITUDE))

# {10}
def main(hdu, measured_data):
    data = hdu[0].data
    download()
    real_data = sort(hdu,max_magnitude=CATALOG_MAGNITUDE_LIMIT, deviation = COLOR_INDEX_DEVIATION_LIMIT, target_color_index = 0.5)
    get_pixs(real_data,hdu,index=2)
    hdu.close()
    mh = match(real_data,measured_data)
    if DEBUG:
        plt.imshow(data,cmap="gray",norm=LogNorm())
        plt.show()
    return mh

# {11}
def loop():
    median, std = None, None
    final = []
    for filename in os.listdir(FOLDER):
        if filename.endswith(".fits"):
            hdu = fits.open(FOLDER+"\\"+filename)
            measured_data, median, std = star_finder(hdu, median, std)
            final += main(hdu, measured_data)
    if not DEBUG:
        med,std = calculate_mean_and_std(final, key = lambda x: x[0]-x[1])
        del_outliers(final, med, std, key = lambda x: x[0]-x[1])

    plot(final)
  
# {12}
def calculate_mean_and_std(mass,key=lambda x: x):
    mass.sort(key=key)
    c = len(mass)
    median = key(mass[c//2])
    sm = 0
    for i in mass:
        xi = key(i)
        sm += (xi-median)**2
    std = sqrt(sm/c)
    return median, std

# {13}
def del_outliers(mass, median, std, key = lambda x: x, kappa = 1.5):
    up = median + kappa*std
    down = median - kappa*std
    for i in range(len(mass)-1, -1,-1):
        val = key(mass[i])
        if val>up or val<down: mass.pop(i)    

# {14}
def plot(comp):
    x = []; y = [];
    for rm, mm, BV, ra_dec, alt in comp:
        if not DEBUG and (mm-rm>1.25+0.243 and alt<1.2 or mm-rm<0+0.243 and alt>1.6): continue
        y.append(mm-rm)
        x.append(alt)
        plt.plot(x[-1],y[-1],'.', color="orange")
        if COLOR_INDEX_ANOTATE: plt.annotate(str(round(BV,2)),(x[-1],y[-1]),color="black")
    print(len(x))
    x = np.array(x).reshape((-1,1))
    y = np.array(y)
    model = LinearRegression().fit(x,y)
    print(model.score(x,y))
    print("k =",model.coef_[0])
    if not DEBUG:
        plt.plot(x,model.predict(x),color="orange")
    plt.xlabel("1/cos(z)")
    plt.ylabel("difference in magnitude")
    plt.show()

if __name__ == "__main__": loop()
