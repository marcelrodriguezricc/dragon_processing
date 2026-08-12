import matplotlib.pyplot as plt
import numpy as np
from cmocean import cm

from .utilities import dragon_select_cast


def plot_beam_variable(data,variable_name,cmap,title=None,vmin=None,vmax=None,with_bottom_track=False):
    fig, axes = plt.subplots(1,data.adcpBeam.size,sharex=True,sharey=True)

    for i,ax in enumerate(axes):
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

    cbar = fig.colorbar(pc,ax=axes)
    cbar.set_label(f'{variable_name} ({data[variable_name].attrs['units']})')

    fig.suptitle(title)
    axes[0].yaxis.set_inverted(True)
    axes[0].set_ylabel('Depth (m)')

    return fig

def plot_gridded_variable(data, variable_name, cmap, label_dim, title=None, vmin=None,vmax=None):
    fig, axes = plt.subplots(1,data[label_dim].size,sharex=True,sharey=True)

    for i,ax in enumerate(axes):
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

    cbar = fig.colorbar(pc,ax=axes)
    cbar.set_label(f'{variable_name} ({data[variable_name].attrs['units']})')

    fig.suptitle(title)
    axes[0].yaxis.set_inverted(True)
    axes[0].set_ylabel('Depth (m)')

    return fig

def initial_dragon_plots(dragon,save_folder,deploy_name):
    beam_vars_to_plot = {
        'beam_correlation' : {
            'vmin' : dragon.adcpCorrelation_cutoff,
            'vmax' : np.nanmax(dragon.beam_correlation),
            'cmap' : cm.turbid_r,
            'variable' : 'beam_correlation'
        },
        'beam_amplitude' : {
            'vmin' : dragon.adcpAmplitude_cutoff,
            'vmax' : np.nanmax(dragon.beam_amplitude),
            'cmap' : cm.amp,
            'variable' : 'beam_amplitude'
        },
        'beam_velocity' : {
            'vmin' : -0.5,
            'vmax' : 0.5,
            'cmap' : cm.balance,
            'variable' : 'velBeam_l1'
        }
    }

    save_folder.mkdir(parents=True,exist_ok=True)

    for cast in dragon.cast:
        cast_down = dragon_select_cast(dragon,cast,cast_type='down')
        cast_up = dragon_select_cast(dragon,cast,cast_type='up')
        for name, var in beam_vars_to_plot.items():
            figs = [
                plot_beam_variable(
                cast_dir,
                var['variable'],
                var['cmap'],
                vmin=var['vmin'],
                vmax=var['vmax'],
                title=f'{deploy_name}_cast_{cast.cast.data}_{dir}'
                )
                for dir,cast_dir in zip(['down','up'],[cast_down,cast_up])
            ]

            figs[0].savefig(save_folder / f'cast_{cast.cast.data}_down_{name}.png')
            figs[1].savefig(save_folder / f'cast_{cast.cast.data}_up_{name}.png')
            plt.close(figs[0])
            plt.close(figs[1])

        figs = [
            plot_gridded_variable(
                cast_dir,
                'velENU_l2',
                cm.balance,
                label_dim='ENU',
                vmin=-0.5,
                vmax=0.5,
                title=f'{deploy_name}_cast_{cast.cast.data}_{dir}'
            )
            for dir,cast_dir in zip(['down','up'],[cast_down,cast_up])
        ]

        figs[0].savefig(save_folder / f'cast_{cast.cast.data}_down_vel_ENU.png')
        figs[1].savefig(save_folder / f'cast_{cast.cast.data}_up_vel_ENU.png')
        plt.close(figs[0])
        plt.close(figs[1])

