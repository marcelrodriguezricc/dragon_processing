import h5py
import numpy as np
import xarray as xr
from .adcp import define_rotation_nortek_XYZ_to_ENU

from .utilities import convert_matlab_datenum, decode_matlab_char, file_save


def get_beam_direction_in_nortek_instrument_coords(beam_angles,degrees=True):
    # Returns the direction aris beams in Nortek instrument coordinates assuming aris is oriented with negative beam angles pointed in x direction and positive beam angles oriented in -x direction.
    # note, this function assumes beam_angle is an array because aris beams are all at different angles.

    if degrees:
        convert = np.pi/180
    else:
        convert = 1

    return np.array([[np.sin(angle*convert),0,np.cos(angle*convert)] for angle in beam_angles])

def get_beam_direction_from_nortek_orientation_earth_coords(beam_angles,heading,roll,pitch,degrees,orientation):
    # Returns the direction of aris beams in Earth coordinates based on the beam angles and instrument orientation, assuming orientation is reported in nortek adcp format.
    # degrees is a boolean that specifies whether the angles are in degrees (True) or radians (False).
    # orientation is a string that specifies whether the nortek instrument is programmed to be oriented 'up' or 'down'.
    # note, this function assumes beam_angle is an array because aris beams are all at different angles.

    beam_directions_instrument = get_beam_direction_in_nortek_instrument_coords(beam_angles,degrees=degrees)
    rotation_to_earth = define_rotation_nortek_XYZ_to_ENU(heading,roll,pitch,degrees=degrees,orientation=orientation)
    beam_directions_earth = np.array([rotation_to_earth.apply(beamDir) for beamDir in beam_directions_instrument])
    return beam_directions_earth

def convert_aris3000(mat_file_in,frequency=3):

    attributes = {
        'beam_width_vertical' : 15,
        'beam_angle_units' : 'degrees',
        'beam_spread' : 30,

    }
    if frequency == 1.8:
        attributes['beam_width_horizontal'] = 0.3
    elif frequency == 3:
         attributes['beam_width_horizontal'] = 0.2     
    else:
        raise ValueError('frequency must be 1.8 or 3 MHz')
    
    ds = xr.Dataset()

    with h5py.File(mat_file_in,'r') as f:
        frame = f['frame']
        header = f['header']
        n_beams = int(np.squeeze(header['numbeams']))
        n_bins = int(np.squeeze(header['sampleperchannel']))

        # ds.coords['distance'] = ('bin',np.linspace(np.squeeze(header['minrange']),np.squeeze(header['maxrange']),int(np.squeeze(header['sampleperchannel']))))
        # ds.distance.attrs['units'] = 'm'
        t0 = convert_matlab_datenum(np.squeeze(header['datenum']))
        dt = ((np.arange(np.squeeze(header['numframes']))/np.squeeze(header['framerate']))*1e9).astype('timedelta64[ns]')
        ds.coords['time'] =  t0 + dt
        ds.coords['beam'] = np.arange(int(n_beams))+1
        ds.coords['bin'] = np.arange(int(n_bins))+1

        ds['min_range'] = ('time',np.full(dt.shape,np.squeeze(header['minrange'])))
        ds.min_range.attrs['units'] = 'm'
        ds['max_range'] = ('time',np.full(dt.shape,np.squeeze(header['maxrange'])))
        ds.max_range.attrs['units'] = 'm'
        ds['number_of_bins'] = ('time',np.full(dt.shape,n_bins))

        data = np.squeeze(frame)
        known_dims = {
            'time' : dt.size,
            'beam' : n_beams,
            'bin' : n_bins
        }
        dims = [next(dim for dim, n in known_dims.items() if n==s) for s in data.shape]
        ds['frame'] = (dims,data)
        # if data.shape[1] == n_beams:
        #     ds['frame'] = (('time','beam','bin'),data)
        # else:
        #     ds['frame'] = (('time','bin','beam'),data)
        ds.coords['beam_angle'] = ('beam',np.linspace(0,30,n_beams)-15)

        attributes.update({
            'version' : int(np.squeeze(header['version'])),
            'serial_number' : int(np.squeeze(header['serialnumber'])),
            'frame_rate' : np.squeeze(header['framerate']),
            'configuration' : decode_matlab_char(header['configuration'][:]),
        })

        ds.attrs = attributes

    ds = ds.transpose('time','bin','beam')

    return ds

def process_deployment_aris(deployment_path,aris_frequency,overwrite=False):

    aris_paths = sorted(deployment_path.rglob('*.aris'))
    aris_paths = [path for path in aris_paths if path.parent.name != 'files_that_are_not_parsing']
    if not aris_paths:
        print('no .aris data found')
        return None
    
    mat_paths = [path.with_suffix('.mat') for path in aris_paths]
    file_base = [path.name.split('_')[0] for path in aris_paths]
    if not np.unique(file_base).size == 1:
        raise ValueError('more than 1 aris file base name found. code cannot handle this')
    file_base = file_base[0]

    nc_file = mat_paths[0].parent / f'{file_base}_aris.nc'
    aris_data_list = []
    if not overwrite and nc_file.is_file():
        print(f'skipping matlab conversion to {nc_file.name}, already converted')
        aris_data = xr.load_dataset(nc_file)
    else:
        for mat_file in mat_paths:
            if not mat_file.is_file():
                print(f'{mat_file.name} does not exist. please process with matlab first')
                return None
            print(f'converting {mat_file.name}')
            aris_data_list.extend([convert_aris3000(mat_file,frequency=aris_frequency),])

        print(f'combining {len(aris_data_list)} files')
        n_bins = np.max([data.bin.size for data in aris_data_list])
        all_bins = np.arange(n_bins)+1
        aris_data = xr.concat([ds.reindex(bin=all_bins) for ds in aris_data_list],dim='time').sortby('time')

        file_save(aris_data,nc_file,overwrite=overwrite)
        
    return aris_data
        

