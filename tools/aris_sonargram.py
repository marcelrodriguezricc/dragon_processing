"""Export ARIS frames from a combined *_aris.nc file as sonar fan images (.png).

Intensity is colored with viridis (0-255). Optional filters, whose filtered-out samples are set to 0
so they draw at the bottom of the viridis scale:
    --threshold: samples at or below this intensity are filtered out
    --min-range/--max-range: samples outside this distance from the sonar are filtered out

Arguments:
    <nc_file>: input NetCDF file with dragon aris3000 data
    <output_dir>: directory to which rendered frames will be written
    --start/--end: start and end frames for visualization
    --start-time/--end-time: start and end datetime timestamp for visualization (UTC), will approximate to closest timestamp
    --min-range/--max-range: specifies minimum and maximum extents for band-pass range filter
    --threshold: specifies high-pass intensity threshold value

Usage:
    python tools/aris_sonargram.py "/Volumes/LeConte Pro/dragon_data/raw/20260817_1800_dragon/2026-08-17_aris.nc" frames/
    python tools/aris_sonargram.py <nc_file> <output_dir> --start 1000 --end 2000
    python tools/aris_sonargram.py <nc_file> <output_dir> --start-time 2026-08-17T20:15:00 --end-time 2026-08-17T20:20:00
    python tools/aris_sonargram.py <nc_file> <output_dir> --start 1000 --end 2000 --threshold 80 --min-range 2 --max-range 8
"""

# ===== Dependencies =====
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # No window is opened, images are only saved
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# ========================

# ===== Helper Functions =====

def edges(centers):
    # Get the edges of each cell, halfway between the center of current cell and the next

    mid = (centers[1:] + centers[:-1]) / 2 # Find midpoint between current cell and next
    return np.concatenate([[2 * centers[0] - mid[0]], mid, [2 * centers[-1] - mid[-1]]]) # Generate the first and last edges, and return that with all other midpoints


def fan_coordinates(min_range, max_range, n_bins, beam_angle):
    # Get list of all cell corners in cartesian coordinates for pcolormesh

    ranges = np.linspace(min_range, max_range, n_bins) # Get range extents
    angles = np.deg2rad(beam_angle) # Get per-beam angles
    r, a = np.meshgrid(edges(ranges), edges(angles), indexing='ij') # Generate a list of cell corners from range (arcs some distance from sonar) and angle (straight rays propagating from sonar at some angle) edges
    return r * np.sin(a), r * np.cos(a) # Convert from polar to cartesian (meters)


def nearest_frame(times, timestamp):
    # Get the index of the frame whose time is closest to the timestamp (e.g. 2026-08-17T20:15:00)

    return int(np.abs(times - np.datetime64(timestamp)).argmin()) # Subtract specified timestamp from all timestamps and get lowest


def read_frame(ds, i, filter_min, filter_max):
    # Get frame i, the distance of each of its bins from the sonar, and which bins fall inside the range filter

    n_bins = int(ds.number_of_bins[i]) # Get number of bins
    image = ds.frame[i, :n_bins, :].values # Drop the zero padding beyond this frame's bins
    ranges = np.linspace(float(ds.min_range[i]), float(ds.max_range[i]), n_bins) # Distance of each bin from the sonar
    in_range = (ranges >= filter_min) & (ranges <= filter_max) # Bins between the range filter cutoffs
    return image, ranges, in_range

# ============================

# ===== Main Runtime Loop =====

def main():

    # Arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('nc_file', type=Path)
    parser.add_argument('output_dir', type=Path)
    start_group = parser.add_mutually_exclusive_group() # Start by frame index or by time, not both
    start_group.add_argument('--start', type=int, default=0, help='first frame to export (default: 0)')
    start_group.add_argument('--start-time', help='first frame to export is the one nearest this time, e.g. 2026-08-17T20:15:00')
    end_group = parser.add_mutually_exclusive_group() # End by frame index or by time, not both
    end_group.add_argument('--end', type=int, help='last frame to export (default: last frame in the file)')
    end_group.add_argument('--end-time', help='last frame to export is the one nearest this time, e.g. 2026-08-17T20:20:00')
    parser.add_argument('--threshold', type=int, help='intensity a sample must exceed to survive the filter (default: no intensity filter)')
    parser.add_argument('--min-range', type=float, default=0, help='closest range from the sonar (m) a sample must be at to survive the filter (default: no limit)')
    parser.add_argument('--max-range', type=float, default=float('inf'), help='farthest range from the sonar (m) a sample must be at to survive the filter (default: no limit)')
    args = parser.parse_args()
    if args.threshold is not None and not 0 <= args.threshold < 255:
        parser.error('--threshold must be between 0 and 254')
    if args.min_range >= args.max_range:
        parser.error('--min-range must be less than --max-range')

    # Load dataset (frames stay on disk and are read one at a time)
    ds = xr.open_dataset(args.nc_file)

    # Get pertinent values from dataset
    times = ds.time.values
    min_range = ds.min_range.values
    max_range = ds.max_range.values
    frame_rate = ds.frame_rate.values
    beam_angle = ds.beam_angle.values

    # Start at specified frame if no start timestamp is specified, or else start the frame nearest to specified start time
    start = args.start if args.start_time is None else nearest_frame(times, args.start_time)
    if args.end_time is not None: # If end time is specified...
        end = nearest_frame(times, args.end_time) # End at frame with timestamp nearest to that specified
    else: # If no end time is specified...
        end = times.size - 1 if args.end is None else min(args.end, times.size - 1) # Check if end frame is specified, if not pick last frame
    if start > end: # Throw an error if the start time is later than the end time
        parser.error(f'start frame ({start}) is after end frame ({end})')
    frames = range(start, end + 1)
    print(f'exporting frames {start}-{end} ({str(times[start])[:23]} to {str(times[end])[:23]})')

    args.output_dir.mkdir(parents=True, exist_ok=True) # Create the output directory if needed
    digits = len(str(times.size - 1)) # Zero-pad file names so they sort in frame order

    # Generate figure with one fixed extent covering every selected frame, so the scale doesn't jump between recordings
    r_min, r_max = min_range[frames].min(), max_range[frames].max() # Get minimum and maximum range
    max_angle = np.deg2rad(np.abs(beam_angle).max()) # Determine extent of furthest angle
    fig, ax = plt.subplots(figsize=(7, 7), layout='constrained')
    ax.set_xlim(-r_max * np.sin(max_angle), r_max * np.sin(max_angle)) # Set x-axis limits
    ax.set_ylim(r_min * np.cos(max_angle), r_max) # Set y-axis limits
    ax.set_aspect('equal') # Set equal aspect
    ax.set_xlabel('Across-beam Distance (m)') # Set x-axis label
    ax.set_ylabel('Range from Sonar (m)') # Set y-axis label
    ax.set_facecolor('white')
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, 255), 'viridis'), ax=ax, shrink=0.7, label='Echo Intensity (0 - 255)') # Generate colorbar

    mesh, window = None, None # Initialize figure
    for i in frames: # For each frame...
        image, _, in_range = read_frame(ds, i, args.min_range, args.max_range) # Get frame, distance, and bins
        n_bins = in_range.size # Set size based on number of bins
        survives = in_range[:, np.newaxis] if args.threshold is None else in_range[:, np.newaxis] & (image > args.threshold) # Samples inside the range filter and, if given, above the threshold
        image = np.where(survives, image, 0) # Set filtered-out samples to 0, the bottom of the viridis scale

        # Sonar fan
        frame_window = (min_range[i], max_range[i], n_bins) # Establish window from frame parameters
        if frame_window == window: # If geometry has been set by the previous frame...
            mesh.set_array(image) # Same fan geometry as the last frame, update pixels only
        else: # If geometry has not been set by the previous frame
            if mesh is not None: # And the mesh has been set...
                mesh.remove() # Remove the previous mesh
            x, y = fan_coordinates(min_range[i], max_range[i], n_bins, beam_angle) # Get cartesian coordinates for each cell from sonar polar
            mesh = ax.pcolormesh(x, y, image, cmap='viridis', vmin=0, vmax=255, shading='flat') # Set mesh
            window = frame_window # Set window for next frame to reference geometry

        ax.set_title(
            f'frame {i} of {times.size - 1}    {str(times[i])[:23]}\n'
            f'{min_range[i]:.1f}-{max_range[i]:.1f} m, {n_bins} bins, {frame_rate[i]:.1f} FPS',
            fontsize=10,
        ) # Set title of figure
        fig.savefig(args.output_dir / f'frame_{i:0{digits}d}.png', dpi=150) # Save frame
        if (i - start) % 100 == 0:
            print(f'Frame {i}/{end}')

    plt.close(fig)
    print(f'Saved {len(frames)} frames to {args.output_dir}')

# =============================

if __name__ == '__main__':
    main()
