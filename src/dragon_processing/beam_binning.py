import numpy as np

from .utilities import (
    dataset_convert_timestamp,
    dragon_select_cast,
    find_nearest_indices,
    least_squares_matrix,
    project_to_line,
    undo_projection,
)


def bin_calc(binX,dx,x,indOK,beamVel,beamNum,beam_directions,beam_backscatter,nBeams=5):

    waterVel = np.full((binX.size,3),np.nan)
    waterVelCov = np.full((binX.size,3,3),np.nan)
    waterVelmin_beam_residual = np.full((binX.size,nBeams),np.nan)
    backscatter = np.full(binX.size,np.nan)
    xInd = np.full(beam_backscatter.shape,np.nan)

    for xi in range(binX.size):
        # Indices for points within current horizontal bin
        indLoc = np.logical_and(x>=binX[xi]-dx,x<binX[xi]+dx)

        # Combine horizontal index with existing mask
        indTotal = np.where(np.logical_and(indOK,indLoc))

        # Extract beam velocities for current bin
        beamVels = beamVel[indTotal]
        iGood = ~np.isnan(beamVels)

        # Get directions and velocities for valid points
        dirTemp = beam_directions[indTotal][iGood,:]
        bVelTemp = beamVels[iGood]
        beamNumTemp = beamNum[indTotal][iGood]

        backscatter[xi] = np.nanmean(beam_backscatter[indTotal][iGood])

        xInd[tuple(ind[iGood] for ind in indTotal)] = xi

        if bVelTemp.size>3:
            # waterVel[xi,:], _, waterVelCov[xi,:,:] = utf.leastSquaresMatrix(bVelTemp,dirTemp)
            waterVel[xi,:], predictedTemp, _ = least_squares_matrix(bVelTemp,dirTemp)
            waterVelCov[xi,:,:] = np.linalg.inv(np.matmul(dirTemp.T,dirTemp))
        
            for bnum in np.unique(beamNumTemp):
                indBeam = beamNumTemp==bnum
                if (np.sum(indBeam)>1) and (np.all(bVelTemp[indBeam]-predictedTemp[indBeam]>0) or np.all(bVelTemp[indBeam]-predictedTemp[indBeam]<0)):
                    waterVelmin_beam_residual[xi,bnum-1] = np.nanmin(np.abs(bVelTemp[indBeam]-predictedTemp[indBeam])/np.nanstd(bVelTemp[indBeam]))
                    


    return waterVel, waterVelCov, waterVelmin_beam_residual, backscatter, xInd


def field_qc(cast,covCutoff=2, beamResidualCutoff=2, maxVel = 1):

    variableNames = [name for name in list(cast.data_vars) if 'field_velocity' in name and (not 'cov' in name and not 'min_beam_residual' in name)]

    #slope = 1/np.tan((cast.mean_heading-90)*np.pi/180)
    #theta = np.pi/2-np.abs(np.atan(slope))

    theta = (cast.mean_heading-90)*np.pi/180

    for name in variableNames:
        cov = cast[name + '_cov']
        vel = cast[name]
        min_beam_residual = cast[name+'min_beam_residual']
    
        uBad = np.logical_or(cov[...,0,0]>covCutoff,vel[...,0]>maxVel)
        vBad = np.logical_or(cov[...,1,1]>covCutoff,vel[...,1]>maxVel)
        wBad = np.logical_or(cov[...,2,2]>covCutoff,vel[...,2]>maxVel)

        beamResidualBad = np.any(min_beam_residual>beamResidualCutoff, axis=-1)

        iBad = np.logical_or(np.logical_or(np.logical_or(uBad,vBad),wBad),beamResidualBad).data[...,np.newaxis].repeat(3,axis=-1)
    
        cast[name].data[iBad] = np.nan

        #velX = np.cos(theta)*cast[name].data[...,1] - np.sin(theta)*cast[name].data[...,0]
        #velY = np.sin(theta)*cast[name].data[...,1] + np.cos(theta)*cast[name].data[...,0]
        E = cast[name].data[...,0]
        N = cast[name].data[...,1]
        velZ = cast[name].data[...,2]

        velX = N*np.sin(theta) - E*np.cos(theta)
        velY = -N*np.cos(theta) - E*np.sin(theta)

        #velY = -(cast[name].data[...,0]*np.sin(theta) + cast[name].data[...,1]*np.cos(theta))
        #velX = cast[name].data[...,1]*np.sin(theta) - cast[name].data[...,0]*np.cos(theta)
        #velZ = cast[name].data[...,2]

        cast[name.replace('enu','xyz')] = (('field_depth','field_x','xyz'),np.stack((velX,velY,velZ),axis=-1))
        cast[name.replace('enu','xyz')].attrs = cast[name].attrs.copy()
        cast[name.replace('enu','xyz')].attrs['description'] = 'Velocity in XYZ coordinates, where X is along ice face (roughly north due to right hand rule), Y is away from the ice face, and Z is vertical. away from ice face direction determined based on mean ADCP heading.'

    cast.attrs['field_covariance_cutoff'] = covCutoff
    cast.attrs['field_min_beam_residual_cutoff'] = beamResidualCutoff

    return cast

def beam_bin(cast,binSizeX=1,binSizeZ=1,covCutoff=2, beamResidualCutoff=2, maxVel = 1,max_heading_variation=5):
    """
    Bin ADCP beam velocities into 2D bins along projected line from ADCP to ice face.
    Args:
        cast: ADCP cast object containing beam data.
        binSizeX: Horizontal bin size (meters).
        binSizeZ: Vertical bin size (meters).
        covCutoff: Covariance threshold for QC.
        beamResidualCutoff: Minimum beam residual threshold for QC.
        maxVel: Maximum velocity threshold for QC.
    Returns:
        Updated cast object with binned velocity fields.
    """
    # Create depth bins from 0 to max beam depth, rounded up to nearest 10 m
    binDepth = np.arange(0,np.ceil(np.max(cast.beam_depth)/10)*10+binSizeZ,step=binSizeZ)

    # Half bin sizes for binning logic
    dx = binSizeX/2
    dz = binSizeZ/2
    
    #region Calculate mean heading and project ADCP locations to a line

    headingsAll = cast.heading_angle.data

    #determine indices for allowable headings by iteratively removing the worst heading until all headings are within allowable distances of original heading
    indGoodHeading = np.ones(headingsAll.size,dtype=bool)
    indGoodHeading[np.isnan(headingsAll)] = False
    while(np.nanmax(np.abs(headingsAll[indGoodHeading]-np.nanmean(headingsAll[indGoodHeading])))>max_heading_variation):  
        indGoodHeading[np.where(indGoodHeading)[0][np.nanargmax(np.abs(headingsAll[indGoodHeading]-np.nanmean(headingsAll[indGoodHeading])))]] = False
    
    heading = np.nanmean(headingsAll[indGoodHeading])

    indGoodHeadingData = indGoodHeading[find_nearest_indices(cast.time_full.data,cast.time.data)]

    # indBadHeading = np.abs((utf.interpDatetime64(cast.time.data,cast.time_full.data,cast.heading_angle.data)-heading+180)%360-180)>100 #catch times where ADCP is just looking the wrong way, but projection distances won't do.
    slope = 1/np.tan((heading-90)*np.pi/180)
    intercept = np.mean(cast.adcp_location[indGoodHeadingData,1].data)-slope*np.mean(cast.adcp_location[indGoodHeadingData,0].data)
    
    # Project beam and wall locations onto the line
    xnew,ynew = project_to_line(x=cast.beam_location[...,0].data,y=cast.beam_location[...,1].data,slope = slope,intercept=intercept)
    xnewWall,ynewWall = project_to_line(x=cast.beam_wall_location[...,0].data,y=cast.beam_wall_location[...,1].data,slope = slope,intercept=intercept)

    indProjOK = indGoodHeadingData[:,np.newaxis,np.newaxis].repeat(cast.dist.size,axis=1).repeat(cast.adcp_beam.size,axis=2)
    indProjOKWall = indGoodHeadingData[:,np.newaxis].repeat(cast.adcp_beam.size,axis=1)
    
    # Reference point (x0, y0) at max y along projected line (for ok projections)
    y0 = np.nanmax(ynewWall[indProjOKWall])
    x0 = (y0-intercept)/slope
    
    # Calculate projected distances from reference point (for all points because we will select ok points later?)
    xProj = np.sqrt((xnew-x0)**2+(ynew-y0)**2)
    xProjWall = np.sqrt((xnewWall-x0)**2+(ynewWall-y0)**2)

    # Adjust sign of projected distances based on position relative to reference point
    xProj[ynew<y0] = -xProj[ynew<y0]
    xProjWall[ynewWall<y0] = -xProjWall[ynewWall<y0]

    #caclulate wall location in each depth bin for each beam
    wallLoc = np.full((binDepth.size,cast.adcp_beam.size),np.nan)

    for de in range(binDepth.size):
        # Indices for points within current depth bin
        indWall = np.logical_and(np.logical_and(-cast.beam_wall_location[...,2]>=binDepth[de]-dz,-cast.beam_wall_location[...,2]<binDepth[de]+dz),indProjOKWall)
        # For each beam, store wall location if present in bin
        for i in range(cast.adcp_beam.size):
            if np.sum(indWall[:,i]>0):
                wallLoc[de,i] = np.nanmean(xProjWall[:,i][indWall[:,i]])

    # Create horizontal bins along projected line
    xmin = np.min(xProj[indProjOK]) #np.min([np.min(xProj),np.min(xProj2)])
    xmax = np.max(xProj[indProjOK]) #np.max([np.max(xProj),np.max(xProj2)])
    binXproj = np.arange(np.floor(xmin/10)*10-binSizeX,np.ceil(xmax/10)*10+binSizeX,step=binSizeX)
    
    # Repeat beam directions for each distance measurement
    beam_directions = np.repeat(dataset_convert_timestamp(cast,variableName='beam_directions_earth',timeFrom='time_full',timeTo='time')[:,np.newaxis,:,:], cast.dist.size, axis=1) 
    
    # Initialize arrays for binned velocities, covariances, wall locations, and backscatter
    waterVel1 = np.full((binDepth.size,binXproj.size,3),np.nan)
    waterVel1Cov = np.full((binDepth.size,binXproj.size,3,3),np.nan)
    waterVel1min_beam_residual = np.full((binDepth.size,binXproj.size,cast.adcp_beam.size),np.nan)
    backscatter = np.full((binDepth.size,binXproj.size),np.nan)
    
    print('starting')
    
    # Extract beam data and valid 3-beam mask
    beamVel = cast.beam_velocity_l2.data
    beamNum = cast.adcp_beam.data[np.newaxis,np.newaxis,:].repeat(cast.time.size,axis=0).repeat(cast.bin.size,axis=1)
    beam_backscatter = cast.beam_backscatter.data
    indQC = np.logical_and(~cast.mask_3beam_beam_velocity.data,cast.mask_beam_cleaning.data)
        
    xInd = np.full(cast.beam_velocity_l2.shape,np.nan)
    dInd = np.full(cast.beam_velocity_l2.shape,np.nan)

    # Loop over depth bins
    for de in range(binDepth.size):
        # Indices for points within current depth bin
        indDepth = np.logical_and(cast.beam_depth>=binDepth[de]-dz,cast.beam_depth<binDepth[de]+dz)
        indOK = np.logical_and(np.logical_and(indDepth,indQC),indProjOK)

        waterVel1[de,...],waterVel1Cov[de,...], waterVel1min_beam_residual[de,...], backscatter[de,:], xInd_temp = bin_calc(binX=binXproj,dx=dx,x=xProj,indOK = indOK,beamVel=beamVel,beamNum=beamNum,beam_directions=beam_directions,beam_backscatter=beam_backscatter,nBeams=cast.adcp_beam.size)
        indIncluded = np.where(~np.isnan(xInd_temp))
        xInd[indIncluded] = xInd_temp[indIncluded]
        dInd[indIncluded] = de

    # Remove all-NaN rows and columns
    non_nan_depths = np.any(~np.isnan(waterVel1),axis=1) #np.logical_or(np.any(~np.isnan(waterVel1), axis=1), np.any(~np.isnan(waterVel2), axis=1))
    non_nan_x = np.any(~np.isnan(waterVel1),axis=0)#np.logical_or(np.any(~np.isnan(waterVel1), axis=0), np.any(~np.isnan(waterVel2), axis=0))
    depth_index = np.where(non_nan_depths)[0]
    x_index = np.where(non_nan_x)[0]

    if depth_index.size>2 and x_index.size>2:
        waterVel1 = waterVel1[depth_index.min():depth_index.max()+1, x_index.min():x_index.max()+1, :]
        waterVel1Cov = waterVel1Cov[depth_index.min():depth_index.max()+1, x_index.min():x_index.max()+1, :, :]
        waterVel1min_beam_residual = waterVel1min_beam_residual[depth_index.min():depth_index.max()+1, x_index.min():x_index.max()+1, :]
        backscatter = backscatter[depth_index.min():depth_index.max()+1, x_index.min():x_index.max()+1]
        wallLoc = wallLoc[depth_index.min():depth_index.max()+1, :]

        dInd[dInd<depth_index.min()] = np.nan
        dInd[dInd>depth_index.max()] = np.nan
        dInd = dInd - depth_index.min()

        xInd[xInd<x_index.min()] = np.nan
        xInd[xInd>x_index.max()] = np.nan
        xInd = xInd - x_index.min()

        binDepth = binDepth[depth_index.min():depth_index.max()+1]
        binXproj = binXproj[x_index.min():x_index.max()+1]

        # Add new coordinates to cast object
        cast.coords['field_depth'] = binDepth
        cast.coords['field_x'] = binXproj
        cast.coords['direction2'] = cast.coords['direction'].copy()
        
        # Store binned velocity and covariance in cast object
        cast['field_velocity_1_enu'] = (('field_depth','field_x','direction'),waterVel1)
        cast.field_velocity_1_enu.attrs = cast.beam_velocity_l2.attrs.copy()
        cast.field_velocity_1_enu.attrs['notes'] = 'Water velocity in East, North, Up coordinates, binned in '+str(binSizeZ)+' m vertical bins and '+str(binSizeX)+' m horizontal bins projected along line from ADCP to ice face based on ADCP heading.'
        cast['field_velocity_1_enu_cov'] = (('field_depth','field_x','direction','direction2'),waterVel1Cov)
        cast.field_velocity_1_enu_cov.attrs['notes'] = 'Covariance matrix of fitted water velocity. Note, this is a covariance scaling (A^TA)^-1 from linear fit, so actual covariance depends on uncertainty of beam velocities.'
        cast['field_velocity_1_enumin_beam_residual'] = (('field_depth','field_x','adcp_beam'),waterVel1min_beam_residual)
        cast.field_velocity_1_enumin_beam_residual.attrs['notes'] = 'Minimum residual magnitude divided by standard devation of beam velocity for each beam with more than 1 velocity used in field velocity fit. If any values are large, this implies that the beam velocities are spread too much or that the fitted velocity is bad at representing this beam suggesting it is observing a different velocity the other beams.'

        cast['field_wall_loc'] = (('field_depth','adcp_beam'),wallLoc)
        cast['field_backscatter'] = (('field_depth','field_x'),backscatter)
        cast.field_backscatter.attrs['notes'] = 'Mean backscatter in each field bin, averaged over all observations used for velocity calculation. Approximation based on amplitude and logarithmic distance decay. Useful only for relative comparisons, not absolute values.'

        cast['beam_field_contribution_depth_index'] = (cast.beam_velocity_l2.dims, dInd)
        cast.beam_field_contribution_depth_index.attrs['description'] = 'Index of the field depth bin that each beam velocity observation contributed to. NaN if observation did not contribute to any field bin.'
        cast['beam_field_contribution_x_index'] = (cast.beam_velocity_l2.dims, xInd)
        cast.beam_field_contribution_x_index.attrs['description'] = 'Index of the field horizontal bin that each beam velocity observation contributed to. NaN if observation did not contribute to any field bin.'

    # Store mean heading in attributes
    cast.attrs['mean_heading'] = np.nanmean(cast.heading_angle)
    cast.attrs['field_intercept'] = intercept
    cast.attrs['field_slope'] = slope
    cast.attrs['field_x0'] = x0
    cast.attrs['field_y0'] = y0

    if depth_index.size>2 and x_index.size>2:
        field_bin_location = np.full((cast.field_depth.size,cast.field_x.size,3),np.nan)
        field_bin_location[...,2] = cast.field_depth.data[:,np.newaxis].repeat(cast.field_x.size,axis=1)
        x,y = undo_projection(cast.field_x.data, cast.field_slope, cast.field_intercept, cast.field_x0, cast.field_y0)
        field_bin_location[...,0] = x[np.newaxis,:].repeat(cast.field_depth.size,axis=0)
        field_bin_location[...,1] = y[np.newaxis,:].repeat(cast.field_depth.size,axis=0)

        cast['field_bin_location'] = (('field_depth','field_x','direction'), field_bin_location)
        cast.field_bin_location.attrs['description'] = 'Coordinates of bin centers in XYZ coordinates.'

        # Run QC and convert to XYZ coordinates
        cast = field_qc(cast,covCutoff=covCutoff, beamResidualCutoff=beamResidualCutoff, maxVel=maxVel)
        cast.attrs['field_calculated'] = True
    else: 
        print('Not enough valid data to store binned velocities.')
        cast.attrs['field_calculated'] = False

    print('done')

    return cast

#endregion

def beam_bin_deployment(deployment,binSizeX=1,binSizeZ=1,covCutoff=2,beamResidualCutoff=2,maxVel = 1,max_heading_variation=60):

    cast_binned_dict = {}

    field_depth = np.empty(0)
    field_x = np.empty(0)
    i = 1
    for cast_num in deployment.cast.data:
        print(f'Binning cast {i} of {deployment.cast.size} (cast number {cast_num})')
        i+=1
        if cast_num % 1 == 0:
            # cast is a downcast
            cast = dragon_select_cast(deployment,cast_num=cast_num,cast_type='down')
        else:
            # cast is an upcast
            cast = dragon_select_cast(deployment,cast_num=int(cast_num),cast_type='up')

        cast_binned_dict[cast_num] = beam_bin(cast,binSizeX=binSizeX,binSizeZ=binSizeZ,covCutoff=covCutoff,beamResidualCutoff=beamResidualCutoff,maxVel=maxVel,max_heading_variation=max_heading_variation)
        if cast_binned_dict[cast_num].attrs['field_calculated']:
            field_depth = np.union1d(field_depth,cast_binned_dict[cast_num].field_depth.data)
            field_x = np.union1d(field_x,cast_binned_dict[cast_num].field_x.data)

    print('Storing results in deployment dataset')


    deployment.coords['field_depth'] = field_depth
    deployment.coords['field_x'] = field_x

    deployment['cast_heading'] = (('cast'), np.array([cast_binned_dict[cn].attrs['mean_heading'] for cn in deployment.cast.data]))
    deployment['field_intercept'] = (('cast'), np.array([cast_binned_dict[cn].attrs['field_intercept'] for cn in deployment.cast.data]))
    deployment['field_slope'] = (('cast'), np.array([cast_binned_dict[cn].attrs['field_slope'] for cn in deployment.cast.data]))
    deployment['field_x0'] = (('cast'), np.array([cast_binned_dict[cn].attrs['field_x0'] for cn in deployment.cast.data]))
    deployment['field_y0'] = (('cast'), np.array([cast_binned_dict[cn].attrs['field_y0'] for cn in deployment.cast.data]))
    deployment.attrs['beam_binning_max_heading_variation'] = max_heading_variation

    variableNames = ['field_velocity_1_enu','field_velocity_1_xyz','field_velocity_1_enu_cov','field_bin_location','field_wall_loc','field_backscatter']#'waterVel2ENU','waterVel2XYZ','waterVel2ENUCov']

    variablesFull = {name: np.full((deployment.cast.size,deployment.field_depth.size,deployment.field_x.size,3,3),np.nan) if 'cov' in name else 
                     np.full((deployment.cast.size,deployment.field_depth.size,deployment.adcp_beam.size),np.nan) if 'wall' in name else
                     np.full((deployment.cast.size,deployment.field_depth.size,deployment.field_x.size),np.nan) if 'backscatter' in name else
                     np.full((deployment.cast.size,deployment.field_depth.size,deployment.field_x.size,3),np.nan) for name in variableNames}
    
    xIndFull = np.full((deployment.time.size,deployment.bin.size,deployment.adcp_beam.size),np.nan)
    dIndFull = np.full((deployment.time.size,deployment.bin.size,deployment.adcp_beam.size),np.nan)
    castIndFull = np.full((deployment.time.size,deployment.bin.size,deployment.adcp_beam.size),np.nan)
    """
    waterVel1ENU_full = np.full((deployment.cast.size,deployment.field_depth.size,deployment.field_x.size,3),np.nan)
    waterVel1XYZ_full = np.full((deployment.cast.size,deployment.field_depth.size,deployment.field_x.size,3),np.nan)
    waterVel1ENUCov_full = np.full((deployment.cast.size,deployment.field_depth.size,deployment.field_x.size,3,3),np.nan)
    waterVel2ENU_full = np.full((deployment.cast.size,deployment.field_depth.size,deployment.field_x.size,3),np.nan)
    waterVel2XYZ_full = np.full((deployment.cast.size,deployment.field_depth.size,deployment.field_x.size,3),np.nan)
    waterVel2ENUCov_full = np.full((deployment.cast.size,deployment.field_depth.size,deployment.field_x.size,3,3),np.nan)
    field_bin_location_full = np.full((deployment.cast.size,deployment.field_depth.size,deployment.field_x.size,3),np.nan)
    """

    for cast_num in deployment.cast.data:
        cast = cast_binned_dict[cast_num]

        if cast.attrs['field_calculated']:
            cast_ind = np.searchsorted(deployment.cast.data,cast_num)
            depth_ind = np.searchsorted(deployment.field_depth.data,cast.field_depth.data)
            x_ind = np.searchsorted(deployment.field_x.data,cast.field_x.data)

            iGrid1 = np.ix_([cast_ind],depth_ind,x_ind,np.arange(3))
            iGrid2 = np.ix_([cast_ind],depth_ind,x_ind,np.arange(3),np.arange(3))
            iGridWall = np.ix_([cast_ind],depth_ind,np.arange(cast.adcp_beam.size))
            iGridBackscatter = np.ix_([cast_ind],depth_ind,x_ind)

            for name in variableNames:
                variablesFull[name][iGridWall if 'wall' in name else iGridBackscatter if 'backscatter' in name else iGrid1 if 'cov' not in name else iGrid2] = cast[name].data
            
            t_ind = deployment.cast_num.data==cast_num
            xIndTemp = cast.beam_field_contribution_x_index.data.copy()
            xIndTemp[~np.isnan(xIndTemp)] = x_ind[xIndTemp[~np.isnan(xIndTemp)].astype(int)]
            xIndFull[t_ind,:,:] = xIndTemp
            dIndTemp = cast.beam_field_contribution_depth_index.data.copy()
            dIndTemp[~np.isnan(dIndTemp)] = depth_ind[dIndTemp[~np.isnan(dIndTemp)].astype(int)]
            dIndFull[t_ind,:,:] = dIndTemp
            full_mask = np.zeros_like(castIndFull,dtype=bool)
            full_mask[t_ind,:,:] = ~np.isnan(dIndTemp)
            castIndFull[full_mask] = cast_ind
        
        """
        waterVel1ENU_full[iGrid1] = cast.field_velocity_1_enu.data
        waterVel1ENUCov_full[iGrid2] = cast.field_velocity_1_enu_cov.data

        waterVel2ENU_full[iGrid1] = cast.field_velocity2ENU.data
        waterVel2ENUCov_full[iGrid2] = cast.field_velocity2ENUCov.data

        field_bin_location_full[iGrid1] = cast.field_bin_location.data
        """

    for name in variableNames:
        deployment[name] = (('cast','field_depth','adcp_beam') if 'wall' in name else ('cast','field_depth','field_x') if 'backscatter' in name else ('cast','field_depth','field_x','direction') if 'cov' not in name else ('cast','field_depth','field_x','direction','direction2'), variablesFull[name])
        deployment[name].attrs = cast[name].attrs.copy()

    deployment['beam_field_contribution_x_index'] = (('time','bin','adcp_beam'), xIndFull)
    deployment['beam_field_contribution_depth_index'] = (('time','bin','adcp_beam'), dIndFull)
    deployment['beam_field_contribution_cast_index'] = (('time','bin','adcp_beam'), castIndFull)

    """
    deployment[f'field_velocity_1_enu'] = (('cast','field_depth','field_x','direction'),waterVel1ENU_full)
    deployment[f'field_velocity_1_enu_cov'] = (('cast','field_depth','field_x','direction','direction2'),waterVel1ENUCov_full)

    deployment[f'field_velocity2ENU'] = (('cast','field_depth','field_x','direction'),waterVel2ENU_full)
    deployment[f'field_velocity2ENUCov'] = (('cast','field_depth','field_x','direction','direction2'),waterVel2ENUCov_full)

    deployment[f'field_bin_location'] = (('cast','field_depth','field_x','direction'),field_bin_location_full)
    """

    return deployment
