from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from scipy.interpolate import interp1d


def file_save(data,file_path,overwrite):
    file_path = Path(file_path)
    if file_path.is_file():
        print(f'file {file_path.name} already exists')
        if overwrite:
            print('overwriting')
            data.to_netcdf(file_path)
        else:
            print('not overwriting, keeping old data')
    else:
        data.to_netcdf(file_path)

def find_nearest_indices(A, B, distanceLimit=None):
    # find the indices of A that are closest to each element in B
    # Ensure both arrays are sorted and 1D
    # if distanceLimit is specified, only return indices where the distance is less than the limit, otherwise return -1 for those indices
    
    A = np.asarray(A)
    idxSort = np.argsort(A)
    Asorted = A[idxSort]
    B = np.asarray(B)
    idx = np.searchsorted(Asorted, B) #returns the indices to insert elements of B into A to maintain order
    idx = np.clip(idx, 1, len(A)-1) # Ensure idx is within bounds of A
    left = Asorted[idx-1] # the value just before the insertion point
    right = Asorted[idx] # the value at the insertion point
    idx_nearest = idxSort[idx - (np.abs(B - left) < np.abs(right - B))] # choose the closest value of left or right

    if distanceLimit is not None:
        idx_nearest[np.abs(B - Asorted[idx_nearest]) > distanceLimit] = -1

    return idx_nearest

def convert_time_to_float(time):
    return (time-time[0]).astype('float')*10**-9  # Convert to seconds since start of time array

def interp_datetime64(x,xp,fp,axis=0,left=None,right=None,fill_value='extrapolate'):
    # Interpolates the values of fp at the times specified by xp to the times specified by x.
    # needed because normal interp functions don't work with datetime64 objects.
    # if fp is multi-dimensional, axis specifies the axis along which to interpolate.

    t0 = np.datetime64('1970-01-01T00:00:00')
    xFloat = (x-t0).astype('timedelta64[ns]').astype('float')
    xpFloat = (xp-t0).astype('timedelta64[ns]').astype('float')

    if fill_value == 'constant':
        f_interp = interp1d(xpFloat, fp, axis=axis, bounds_error=False, fill_value=(left, right))
    elif fill_value == 'extrapolate':
        f_interp = interp1d(xpFloat, fp, axis=axis, fill_value='extrapolate')
    else:
        raise ValueError("fill_value must be 'constant' or 'extrapolate' currently")
    
    return f_interp(xFloat)

def dataset_convert_timestamp(dataset,variableName,timeFrom='time_full',timeTo = 'time'):
    if variableName not in dataset.data_vars:
        raise ValueError(f"Variable '{variableName}' not found in dataset.")

    if timeFrom not in dataset[variableName].coords:
        raise ValueError(f"Time variable '{timeFrom}' not found in variable coordinates.")
    
    if timeTo not in dataset.coords:
        raise ValueError(f"Time variable '{timeTo}' not found in dataset coordinates.")
    
    return interp_datetime64(dataset[timeTo].data,dataset[timeFrom].data,dataset[variableName].data,axis=0)

def lat_lon_to_utm(
    latitude,
    longitude,
    x0=0,
    y0=0
):
    """
    Convert latitude/longitude to UTM coordinates.

    Zone is determined automatically from longitude.

    Returns
    -------
    x : ndarray
    y : ndarray
    zone : int
    """

    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)

    mean_lon = np.nanmean(longitude)
    mean_lat = np.nanmean(latitude)

    zone = int(
        np.floor((mean_lon + 180) / 6) + 1
    )

    epsg = (
        32600 + zone
        if mean_lat >= 0
        else 32700 + zone
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        CRS.from_epsg(epsg),
        always_xy=True,
    )

    x, y = transformer.transform(
        longitude,
        latitude,
    )

    return x-x0, y-y0, zone

def dragon_select_cast(dragon, cast_num=1, cast_type = None):
    # cast_num can be an integer, half integer, or a list of integers and half integers
    # cast_type can be 'down', 'up', 'both', or None

    if isinstance(cast_num, (list, np.ndarray)):
        #if selecting multiple casts

        if cast_type in ['down', 'up']:
            raise ValueError("When selecting multiple casts, cast_type must be 'both' or None")
        
        elif cast_type is None:
            if np.any(~np.logical_or(cast_num%1==0,cast_num%1==0.5)):
                raise ValueError("When selecting multiple casts without specifying cast_type, all elements of cast_num must be integers or half-integers representing downcasts and upcasts respectively.")
            else:
                dragon = dragon.isel({dim: (dragon[dim.replace('time','cast_num')].isin(cast_num)) for dim in dragon.dims if 'time' in dim})

        elif cast_type == 'both':
            if np.any(cast_num%1!=0):
                raise ValueError("When selecting multiple casts with cast_type 'both', all elements of cast_num must be integers.")
            else:
                #select for all casts in cast list (both up and down) along each time dimension
                dragon = dragon.isel({dim: (dragon[dim.replace('time','cast_num')].isin(cast_num) | dragon[dim.replace('time','cast_num')].isin([i + 0.5 for i in cast_num])) for dim in dragon.dims if 'time' in dim})
        else:
            raise ValueError("cast_type must be 'down', 'up', 'both', or None.")

    else:
        #if selecting a single cast
        if cast_type == 'up':
            cast_num = int(cast_num) + 0.5
        elif cast_type == 'down' or cast_type == 'both':
            cast_num = int(cast_num)
        elif cast_type is not None:
            raise ValueError("cast_type must be 'down', 'up', 'both', or None.")
        
        #check that cast_num is present in dataset
        if cast_num not in dragon.cast_num.data:
            raise ValueError(f"cast_num {cast_num} not found in dragon dataset.")

        #select for cast along each time dimension
        if cast_type == 'both':
            for dim in dragon.dims:
                if 'time' in dim:
                    dragon = dragon.isel({dim: (dragon[dim.replace('time','cast_num')] == cast_num) | (dragon[dim.replace('time','cast_num')] == cast_num + 0.5)})
        else:
            for dim in dragon.dims:
                if 'time' in dim:
                    dragon = dragon.isel({dim: dragon[dim.replace('time','cast_num')] == cast_num})

    if 'cast' in dragon.dims:
        #if cast is a dimension (i.e. if multiple casts are present), select for only the casts that are kept
        dragon = dragon.isel({'cast': dragon['cast'].isin(dragon.cast_num.data)})
    if 'deployment' in dragon.dims:
        #if deployment is a dimension (i.e. if multiple deployments are present), select for only the deployments that are kept
        dragon = dragon.isel({'deployment': dragon['deployment'].isin(dragon.deploymentNum.data)})

    return dragon.squeeze()

def amplitude_to_backscatter(amp, distance,alpha = 9.793e-3, convertAmpToDB=False):
    # Relevant references are Deines (1999) and Mullison (2017)
    # the full equation is:
    # 𝑆𝑣 = 𝐶 + 10 log10((𝑇𝑥 + 273.16)𝑅^2) − 𝐿𝐷𝐵𝑀 − 𝑃𝐷𝐵𝑊 + 2𝛼𝑅 +10 log10(10^(𝑘𝑐(𝐸−𝐸𝑟)/10) − 1)
    # where Sv is the volume backscattering that we want, and R is the along beam distance to measurement
    # C, Tx, LDBM, and PDBW, are all constants or instrument specific values
    # 10 ln(10^(𝑘𝑐(𝐸−𝐸𝑟)/10) − 1) is a complicated factor where kc converts counts to decibels,
    # E is the return signal strength indicator amplitude of the ADCP, and Er is a constant representing the noise floor
    # the true purpose of this term is to represent 10log10(S/N), where S is the signal of the ADCP and N is the noise.
    # We don't need true backscatter, we just want to correct for distance related effects, so
    # we can ignore additive effects, simplifying the equation to:
    # Sv = constants + 2𝛼𝑅 + 20 log10(R) + 10log10(amp)
    # the linear term is the absorption of sound in water. 
    #   distance is distance from transducer to scatterer (i.e., depth + range)
    #   the factor of 2 is to account for the two-way travel of the sound
    #   alpha is the absorption coefficient of sound in water (dB/m), which I roughly determined based on http://resource.npl.co.uk/acoustics/techguides/seaabsorption/
    # the log10(R) term is the loss due to the spreading of the beam as it travels away from the transducer. I think this comes from a generic sonar equation and does not depend on instrument or water.
    # the 10log10(amp) term is the conversion from amplitude to decibels, which is likely already done by the instrument and not necessary

    #input amp is amplitude reported by ADCP
    # input convertAmpToDB is a boolean that specifies whether to take 10log10 of amp (True) or not (False)
    # For the adcp I am using, it very clearly needs to be False (it will be obvious if you try both), and I think this is because the amplitude is already in decibels.

    #input distance is distance from transducer to scatterer in meters (along beam!!!!!!)

    #input alpha is the absorption coefficient of sound in water in dB/m - check http://resource.npl.co.uk/acoustics/techguides/seaabsorption/

    #output backscatter is relative and is unitless because we are ignoring all additive constants, so the actual values are meaningless beyond relative differences

    if convertAmpToDB:
        backscatter = 10*np.log10(amp) + 2 * alpha * distance + 20 * np.log10(distance)
    else:
        backscatter = amp + 2 * alpha * distance + 20 * np.log10(distance)

    return backscatter

def project_to_line(x,y,slope,intercept):
    xproj = (y+x/slope-intercept)/(slope+1/slope)
    yproj = slope*xproj+intercept
    return xproj,yproj

def undo_projection(xProj,slope,intercept,x0,y0):

    if isinstance(slope,xr.DataArray):
        slope = slope.data
    if isinstance(intercept,xr.DataArray):
        intercept = intercept.data
    if isinstance(x0,xr.DataArray):
        x0 = x0.data
    if isinstance(y0,xr.DataArray):
        y0 = y0.data
    
    b = -x0+slope*(intercept-y0)
    a = 1+slope**2
    c = x0**2 + (intercept-y0)**2 - xProj**2

    x = np.array([(-b+np.sqrt(b**2 - a*c[i]))/a if xProj[i]>=0 else (-b - np.sqrt(b**2 - a*c[i]))/a for i in range(xProj.size)])

    y = slope*x + intercept

    return x,y

def least_squares_matrix(y,A,dataCov=None):
    #note, this is for solving the equation y = Am, which is a matrix representation of 
    #y_1 = m_1*A_11 + m_2*A_12 + ...
    #y should be entered as a 1 dimensional array of length alpha and will be transformed to have dimensions (alpha,1)
    #A should be a matrix with dimensions (alpha,beta)
    #m has dimensions (beta,1) and will be output as a 1 dimesnional array with length beta.
    #dataCov is the covariance matrix of y, which has dimensions (alpha,alpha).
    #If data in y are uncorrelated, dataCov will be a diagonal matrix with variances of each entry in y on the diagonal.
    #If dataCov is None, assume that dataCov is the identity matrix multiplied by the variance of the fit residuals.

    y = y[:,np.newaxis]

    G = np.matmul(np.linalg.inv(np.matmul(A.T,A)),A.T)

    m = np.matmul(G,y)

    estimate = np.matmul(A,m)

    if dataCov is None:
        error = y-estimate
        std = np.std(error)

        mCov = np.matmul(G,G.T)*std**2
    else:
        mCov = np.matmul(np.matmul(G,dataCov),G.T)

    return np.squeeze(m), np.squeeze(estimate), mCov
