import matplotlib.pyplot as plt
import matplotlib.widgets as mwg
import matplotlib.image as mpimg
import numpy as np

# Homemade module
from imgseries import ImgSeries
from .imbibitionfront_tracking import ImbibitionTracking
import imgbasics


def plot_contour_evolution(data, path='.'):
    """Plot both droplet and the contour tracking as well as the resulting curve

    INPUT:
    - data (panda table): panda table extracted from the class Isotherm
    - path where the images are located

    EXAMPLE:
    s = plot_glevelevolution(data, cycles=[1])
    """

    folder = data['folder'].values
    directory = path + '/' + folder
    images_raw = ImgSeries(paths=directory)
    imbt2 = ImbibitionTracking(paths=directory)
    imbt2.imbibition.load()
    img_ref = imbt2.imbibition.data_image['img_ref']
    crop = imbt2.imbibition.data_image['crop']
    n_start = imbt2.imbibition.data_image['number start']


    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1,
                                   constrained_layout=False, figsize=(12, 8))

    x0, y0 = data['centerx'].values, data['centery'].values
    theta = theta = np.arange(0, 360, 0.5)
    nums, t_unix, radi = data.index.values, data['time (unix)'].values, data['radi'].values
    time_60 = (t_unix - t_unix[0]) / 60

    ax2.grid()
    ax2.plot(time_60, radi, '.-', alpha=0.5)
    ax2.set_xlabel('time (min)')
    ax2.set_ylabel('radius (pixel)')

    img0 = imgbasics.imcrop((images_raw.read(n_start) - img_ref) / img_ref, crop)
    ax1.imshow(img0, vmin=-0.1, vmax=0.2)

    axcolor = 'lightgoldenrodyellow'
    aximg = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)
    slider = mwg.Slider(aximg, 'Image number', valmin=n_start, valinit=n_start,
                        valmax=len(nums) - 1, valstep=1,
                        color='b', alpha=0.3)

    def update(lnum):
        ax1.clear()
        ax2.clear()
        ax2.grid()
        ax2.plot(time_60[lnum == nums], radi[lnum == nums], 'kd')
        ax2.plot(time_60, radi, '.-', alpha=0.5)
        ax2.set_xlabel('time (min)')
        ax2.set_ylabel('radius (pixel)')

        img = imgbasics.imcrop((images_raw.read(lnum) - img_ref) / img_ref, crop)
        x = x0[lnum == nums] + radi[lnum == nums] * np.cos(theta)
        y = y0[lnum == nums] + radi[lnum == nums] * np.sin(theta)
        ax1.imshow(img, vmin=-0.1, vmax=0.2)
        ax1.plot(x, y, '.', markersize=1, linewidth=1)
        ax1.plot(x0[lnum == nums], y0[lnum == nums], 'r+')

    slider.on_changed(update)

    return slider

