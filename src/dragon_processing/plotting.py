import matplotlib.pyplot as plt
import numpy as np
from cmocean import cm
from matplotlib import gridspec

from .utilities import dragon_select_cast


def plot_beam_variable(data,variable_name,cmap,title=None,vmin=None,vmax=None,with_bottom_track=False):

    n_beams = data.adcp_beam.size

    fig = plt.figure(figsize=(9,2),dpi=130,layout='constrained')
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0, hspace=0.0)
    gs = gridspec.GridSpec(1,n_beams+1,width_ratios = [1,]*n_beams+[0.1,],wspace=0,figure=fig)

    profile_axes = [fig.add_subplot(gs[0])]
    profile_axes += [fig.add_subplot(gs[i],sharey=profile_axes[0]) for i in range(1,n_beams)]
    cbar_ax = fig.add_subplot(gs[n_beams])

    for i,ax in enumerate(profile_axes):
        pc=ax.tripcolor(
            data.bin.data[np.newaxis,:].repeat(data.time.size,axis=0).flatten(),
            data.beam_depth[:,:,i].data.flatten(),
            data[variable_name].data[:,:,i].flatten(),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap
            )

        if with_bottom_track:
            ax.plot(data.bottom_distanceBeam,data.bottom_track_depth.data[:,i],color='k')

        ax.set_title(f'Beam {i+1}')
        ax.set_xlabel('Distance (m)')
        ax.axis('equal')

    cbar = plt.colorbar(pc,cax=cbar_ax)
    cbar.set_label(f'{variable_name} ({data[variable_name].attrs['units']})')

    fig.suptitle(title)
    profile_axes[0].yaxis.set_inverted(True)
    profile_axes[0].set_ylabel('Depth (m)')

    return fig

def plot_gridded_variable(data, variable_name, cmap, label_dim, title=None, vmin=None,vmax=None):

    n_profiles = data[label_dim].size

    fig = plt.figure(figsize=(9,3),dpi=130,layout='constrained')
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0, hspace=0.0)
    gs = gridspec.GridSpec(1,n_profiles+1,width_ratios = [1,]*n_profiles+[0.1,],wspace=0,figure=fig)

    profile_axes = [fig.add_subplot(gs[0])]
    profile_axes += [fig.add_subplot(gs[i],sharey=profile_axes[0]) for i in range(1,n_profiles)]
    cbar_ax = fig.add_subplot(gs[n_profiles])

    for i,ax in enumerate(profile_axes):
        pc=ax.pcolormesh(
            data.dist.data,
            data.depth.data,
            data[variable_name].data[:,:,i],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap
        )
        ax.set_title(data[label_dim].data[i])
        ax.set_xlabel('Distance (m)')
        ax.axis('equal')

    cbar = plt.colorbar(pc,cax=cbar_ax)
    cbar.set_label(f'{variable_name} ({data[variable_name].attrs['units']})')

    fig.suptitle(title)
    profile_axes[0].yaxis.set_inverted(True)
    profile_axes[0].set_ylabel('Depth (m)')

    return fig

def plot_field(data,coordinate_system,title=None, vmin_vel=-0.3,vmax_vel=0.3, vmin_scatter=None,vmax_scatter=None, cmap_vel=cm.balance,cmap_scatter = cm.turbid):

    fig = plt.figure(figsize=(9,2),dpi=130,layout='constrained')
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0, hspace=0.0)
    gs = gridspec.GridSpec(1,6,width_ratios = [1,1,1,1,0.1,0.1],wspace=0,figure=fig)

    profile_axes = [fig.add_subplot(gs[0])]
    profile_axes += [fig.add_subplot(gs[i],sharey=profile_axes[0]) for i in range(1,4)]
    cbar_axes = [fig.add_subplot(gs[i]) for i in range(4,6)]

    if coordinate_system == 'ENU':
        variable_name = 'field_velocity_1_enu'
        labels = ['East','North','Vertical']
    elif coordinate_system == 'XYZ':
        variable_name = 'field_velocity_1_xyz'
        labels = ['Along-Ice', 'Cross-Ice','Vertical']
    else:
        raise ValueError('coordinate_system must be ENU or XYZ')

    for i,ax in enumerate(profile_axes[:-1]):
        pc_vel = ax.pcolormesh(
            data.field_x.data,
            data.field_depth.data,
            data[variable_name].data[:,:,i],
            vmin=vmin_vel,
            vmax=vmax_vel,
            cmap=cmap_vel
        )
        ax.set_title(labels[i])
        ax.set_xlabel('Distance (m)')
        ax.axis('equal')

    pc_scatter = profile_axes[-1].pcolormesh(
        data.field_x.data,
        data.field_depth.data,
        data['field_backscatter'].data,
        vmin=vmin_scatter,
        vmax=vmax_scatter,
        cmap=cmap_scatter
    )

    max_depth = np.nanmax(data.depth)+5

    profile_axes[-1].set_title('Backscatter')
    profile_axes[-1].set_xlabel('Distance (m)')
    profile_axes[-1].axis('equal')

    plt.colorbar(pc_vel,cax=cbar_axes[0],location='right').set_label('Velocity (m/s)',labelpad=2)
    plt.colorbar(pc_scatter,cax=cbar_axes[1],location='right').set_label('Relative Backscatter',labelpad=2)

    cbar_vel = plt.colorbar(pc_vel,cax=cbar_axes[0])
    cbar_vel.set_label('Velocity (m/s)')

    cbar_scatter = plt.colorbar(pc_scatter,cax=cbar_axes[1])
    cbar_scatter.set_label('Backscatter')

    fig.suptitle(title)
    # profile_axes[0].yaxis.set_inverted(True)
    profile_axes[0].set_ylim([max_depth+2,0])
    profile_axes[0].set_ylabel('Depth (m)')

    return fig

def initial_dragon_plots(dragon,save_folder,deploy_name):
    beam_vars_to_plot = {
        'beam_correlation' : {
            'vmin' : dragon.adcp_correlation_cutoff,
            'vmax' : np.nanmax(dragon.beam_correlation),
            'cmap' : cm.turbid_r,
            'variable' : 'beam_correlation'
        },
        'beam_amplitude' : {
            'vmin' : dragon.adcp_amplitude_cutoff,
            'vmax' : np.nanmax(dragon.beam_amplitude),
            'cmap' : cm.amp,
            'variable' : 'beam_amplitude'
        },
        'beam_velocity' : {
            'vmin' : -0.3,
            'vmax' : 0.3,
            'cmap' : cm.balance,
            'variable' : 'beam_velocity_l2'
        }
    }

    field_backscatter_min = np.nanmin(dragon.field_backscatter)
    field_backscatter_max = np.nanmax(dragon.field_backscatter)

    save_folder.mkdir(parents=True,exist_ok=True)

    for cast_num in dragon.cast.data:
        cast = dragon_select_cast(dragon,cast_num)
        for name, var in beam_vars_to_plot.items():
            fig = plot_beam_variable(
                cast,
                var['variable'],
                var['cmap'],
                vmin=var['vmin'],
                vmax=var['vmax'],
                title=f'{deploy_name}_cast_{cast_num}'
            )

            fig.savefig(save_folder / f'cast_{cast_num}_{name}.png')
            plt.close(fig)

        fig = plot_gridded_variable(
            cast,
            'velENU_l2',
            cm.balance,
            label_dim='ENU',
            vmin=-0.5,
            vmax=0.5,
            title=f'{deploy_name}_cast_{cast_num}'
        )

        fig.savefig(save_folder / f'cast_{cast_num}_vel_ENU.png')
        plt.close(fig)

        fig = plot_field(
            cast,
            'ENU',
            title=f'{deploy_name}_cast_{cast_num}',
            vmin_vel= -0.3,
            vmax_vel=0.3,
            vmin_scatter=field_backscatter_min,
            vmax_scatter=field_backscatter_max,
            cmap_vel = cm.balance,
            cmap_scatter = cm.turbid
        )
        fig.savefig(save_folder / f'cast_{cast_num}_field_ENU.png')
        plt.close(fig)

        fig = plot_field(
            cast,
            'XYZ',
            title=f'{deploy_name}_cast_{cast_num}',
            vmin_vel= -0.3,
            vmax_vel=0.3,
            vmin_scatter=field_backscatter_min,
            vmax_scatter=field_backscatter_max,
            cmap_vel = cm.balance,
            cmap_scatter = cm.turbid
        )
        fig.savefig(save_folder / f'cast_{cast_num}_field_XYZ.png')
        plt.close(fig)

