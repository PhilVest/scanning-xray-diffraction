from __future__ import print_function
import numpy as np
from skimage.transform import iradon, radon
from skimage.segmentation import random_walker
import sys
import matplotlib.pyplot as plt


def FBP_slice(grains, flt, omegastep, rcut, ymin, ystep, number_y_scans, mode, recon_weights=None):
    grain_masks=[]
    grain_recons=[]
    for i,g in enumerate(grains):
        sinoangles, sino, recon = FBP_grain( g, flt, \
                    ymin, ystep, omegastep, number_y_scans, recon_weights)
        normalised_recon = recon/recon.max()
        grain_recons.append(normalised_recon)

        if mode == 'gradients':
            markers = np.zeros(normalised_recon.shape, dtype=int)
            markers[normalised_recon < rcut[0]] = 1
            markers[normalised_recon > rcut[1]] = 2

            mask = random_walker(normalised_recon, markers, beta=1e4, mode='bf') == 2

            #fig,ax = plt.subplots(1,2, figsize=(15,5))
            #im = ax[0].imshow(mask,aspect="auto")
            #fig.colorbar(im,ax=ax[0])
            #im = ax[1].imshow(normalised_recon,aspect="equal")
            #fig.colorbar(im,ax=ax[1])
            #plt.show()

        # Suggested alternative method for performing the segmentations. It has not been used yet since "gradients"
        # yielded better results for the test data set. Might be useful for other data sets.

        elif mode == 'mass':
            mass = 0
            mask = np.zeros(recon.shape, dtype=bool)
            r = recon.copy()
            for _ in range(1000*1000):
                ii, jj, maxpix = _find_maxpix(mask, r)
                if mass+maxpix>rcut: break
                mask[r==maxpix] = True
                r[r==maxpix] = 0
                mass += maxpix

        # mask = normalised_recon > rcut # Previous method for performing the segmentation, simply by using a threshold.

        grain_masks.append(mask)
    update_grainshapes(grain_recons,grain_masks)
    
    if 0:
        fig,ax = plt.subplots(1,2, figsize=(15,5))
        im = ax[0].imshow(np.sum(grain_masks, axis=0),aspect="auto")
        fig.colorbar(im,ax=ax[0])
        im = ax[1].imshow(np.sum(grain_recons, axis=0),aspect="equal")
        fig.colorbar(im,ax=ax[1])
        plt.show()

    return grain_masks


def FBP_grain( g, flt, ymin, ystep, omegastep, number_y_scans, recon_weights=None):
    """
    Reconstruct a 2d grain shape from diffraction data using Filtered-Back-projection.
    """
    if recon_weights is None:
        recon_weights = [1.0, 1.0]

    # Measured relevant data for the considered grain
    omega = flt.omega[ g.mask ].copy()
    dty = flt.dty[g.mask].copy()
    pixels_per_peak = flt.pixels_per_peak[g.mask].copy()
    sum_intensity = flt.sum_intensity[g.mask].copy()

    if np.min(omega)<0 and np.max(omega)>0:
        # Case 1: -180 to 180 rotation and positive y-scans we make
        # a transformation back into omega=[0 180] in three steps:

        # (I) Half the intensity for peaks entering the sinogram twice.
        doublets_mask = dty<=np.abs(np.min(dty))
        pixels_per_peak[doublets_mask] = pixels_per_peak[doublets_mask]*(1/2)
        sum_intensity[doublets_mask] = sum_intensity[doublets_mask]*(1/2)

        # (II) Map the negative omega values back to positive values, [0 180]
        omega_mask = omega < 0
        omega[omega_mask] = omega[omega_mask] + 180

        # (III) Flip the sign of the y-scan coordinates with negative omega values.
        dty[omega_mask]   = -dty[omega_mask]

    elif np.min(omega)>=0 and np.max(omega)<=180:
        # Case 2: 0 to 180 rotation and both negative and positive y-scans
        # nothing needs be done since this is the standrad tomography case
        pass
    else:
        raise ValueError("Scan configuration not implemented. omega=0,180 scans with positive and \
                          negative y-translations and omega=-180,180 scans with all positive      \
                          y-translations are supported.")

    # Angular range into which to bin the data (step in sinogram)
    angular_bin_size = 180./(number_y_scans)

    # Indices in sinogram for the y-scan and angles
    iy = np.round( (dty - ymin) / ystep ).astype(int)
    iom = np.round( omega / angular_bin_size ).astype(int)

    recons = [None, None]
    for i, measure in enumerate([pixels_per_peak, sum_intensity]):
        # Build the sinogram by accumulating intensity
        sinogram = np.zeros( ( number_y_scans, np.max(iom)+1 ), np.float )
        for j,I in enumerate(measure):
            dty_index   = iy[j]
            omega_index = iom[j]
            sinogram[dty_index, omega_index] += I

        # Normalise the sinogram to account for the intensity not being proportional
        # only to density but also to eta and theta and a lot of other stuff.
        normfactor = sinogram.max(axis=0)
        #normfactor = np.sum(sinogram, axis=0)
        normfactor[normfactor==0]=1.0
        sinogram = sinogram/normfactor

        # Perform reconstruction by inverse radon transform of the sinogram
        theta = np.linspace( angular_bin_size/2., 180. - angular_bin_size/2., sinogram.shape[1] )
        back_projection = iradon( sinogram, theta=theta, output_size=number_y_scans, circle=True )
        back_projection = back_projection/back_projection.max()
        recons[i] = back_projection * recon_weights[i]
    
    recon = np.sum(recons, axis=0)

    if 0:
        fig,ax = plt.subplots(1,2, figsize=(15,5))
        im = ax[0].imshow(sinogram,aspect="auto")
        ax[0].set_title("Sinogram")
        ax[0].set_xlabel(r"$\omega$ bins ["+str(np.round(angular_bin_size,3))+r"$^o$]")
        ax[0].set_ylabel(r"$\Delta y$ index (step="+str(ystep)+r"$\mu$ m)")
        fig.colorbar(im,ax=ax[0])
        im = ax[1].imshow(back_projection,aspect="equal")
        fig.colorbar(im,ax=ax[1])
        ax[1].set_title("Backprojection")
        plt.show()

    return [], sinogram, recon

# def FBP_grain( g, flt, ymin, ystep, omegastep, number_y_scans ):
#     """
#     Reconstruct a 2d grain shape from diffraction data using Filtered-Back-projection.
#     """

#     # Measured relevant data for the considered grain
#     omega = flt.omega[ g.mask ].copy()
#     dty = flt.dty[g.mask].copy()
#     sum_intensity = flt.sum_intensity[g.mask].copy()

#     if np.min(omega)<0 and np.max(omega)>0:
#         # Case 1: -180 to 180 rotation and positive y-scans we make
#         # a transformation back into omega=[0 180] in three steps:

#         # (I) Half the intensity for peaks entering the sinogram twice.
#         doublets_mask = dty<=np.abs(np.min(dty))
#         sum_intensity[doublets_mask] = sum_intensity[doublets_mask]*(1/2)

#         # (II) Map the negative omega values back to positive values, [0 180]
#         omega_mask = omega < 0
#         omega[omega_mask] = omega[omega_mask] + 180

#         # (III) Flip the sign of the y-scan coordinates with negative omega values.
#         dty[omega_mask]   = -dty[omega_mask]

#     elif np.min(omega)>=0 and np.max(omega)<=180:
#         # Case 2: 0 to 180 rotation and both negative and positive y-scans
#         # nothing needs be done since this is the standrad tomography case
#         pass
#     else:
#         raise ValueError("Scan configuration not implemented. omega=0,180 scans with positive and \
#                           negative y-translations and omega=-180,180 scans with all positive      \
#                           y-translations are supported.")

#     keys = np.array([ g.hkl[0], g.hkl[1], g.hkl[2], g.etasigns ]).T
#     unique_reflections = np.unique(keys,axis=0)
#     number_of_reflections = len(unique_reflections)
#     sinogram = np.zeros( ( number_y_scans, number_of_reflections ), np.float )
#     angles = np.zeros( ( number_of_reflections, ), np.float )
#     for reflection_index,reflection in enumerate( unique_reflections ):
#         # h==h, k==k, l==l, eta sign==eta sign
#         mask = (keys == reflection).astype(int).sum(axis=1) == 4
#         for y,om,I in zip(dty[mask],omega[mask],sum_intensity[mask]):
#             dty_index   = np.round( (y - ymin) / ystep ).astype(int)
#             sinogram[dty_index, reflection_index] += I
#         angles[reflection_index] = np.mean(omega[mask])

#     # Sort sinogram in increasing angular order.
#     indx = np.argsort(angles)
#     angles = angles[indx]
#     sinogram = sinogram[:,indx]

#     # Normalise the sinogram to account for the intensity not being proportional
#     # only to density but also to eta and theta and a lot of other stuff.
#     normfactor = sinogram.max(axis=0)
#     normfactor[normfactor==0]=1.0
#     sinogram = sinogram/normfactor

#     # Perform reconstruction by inverse radon transform of the sinogram
#     back_projection = iradon( sinogram, theta=angles, output_size=number_y_scans, circle=True )

#     print(angles)
#     fig,ax = plt.subplots(1,2, figsize=(15,5))
#     im = ax[0].imshow(sinogram,aspect="auto")
#     ax[0].set_title("Sinogram")
#     ax[0].set_xlabel("Unique hkl reflections by increasing omega")
#     ax[0].set_ylabel("dty index (step="+str(ystep)+r"$\mu$ m)")
#     fig.colorbar(im,ax=ax[0])
#     im = ax[1].imshow(back_projection,aspect="equal")
#     fig.colorbar(im,ax=ax[1])
#     ax[1].set_title("Backprojection")
#     plt.show()
#     return [], sinogram, back_projection 

def update_grainshapes( grain_recons, grain_masks):
    '''
    Update a set of grain masks based on their overlap and intensity.
    At each point the grain with strongest intensity is assigned

    Assumes that the grain recons have been normalized
    '''

    for i in range(grain_recons[0].shape[0]):
        for j in range(grain_recons[0].shape[1]):
            if conflict_exists(i,j,grain_masks):
                max_int = 0.0
                leader  = None
                for n,grain_recon in enumerate(grain_recons):
                    if grain_recon[i,j]>max_int:
                        leader = n
                        max_int = grain_recon[i,j]

                #The losers are...
                for grain_mask in grain_masks:
                    grain_mask[i,j]=0

                #And the winner is:
                grain_masks[leader][i,j]=1

def conflict_exists( i,j,grain_masks):
    '''
    Help function for update_grainshapes()
    Checks if two grain shape masks overlap at index i,j
    '''
    claimers = 0
    for grain_mask in grain_masks:
        if grain_mask[i,j]==1:
            claimers+=1
    if claimers>=2:
        return True
    else:
        return False

def _find_maxpix(mask, recon):
    if np.sum(mask)==0:
        i,j = np.where(recon==np.max(recon))
    else:
        vals, idx = [], []
        for i in range(1,mask.shape[0]-1):
            for j in range(1,mask.shape[1]-1):
                if not mask[i,j] and (mask[i-1,j] or mask[i,j-1] or mask[i+1,j] or mask[i,j+1]):
                    vals.append(recon[i,j])
                    idx.append( (i,j)  )
        i,j  = idx[np.argmax(vals)]
    return i, j, recon[i,j]

