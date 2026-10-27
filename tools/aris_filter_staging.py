"""Export ARIS frames as .png images showing the sonargram next to that frame's intensity and range histograms.

Samples at or below --threshold are filtered out and drawn black. Samples that survive are colored
on a blue-to-red gradient, from just above the threshold (blue) to 255 (red).
Samples outside --min-range/--max-range are filtered out entirely: they are gray in the fan and
left out of the intensity histogram.

The range histogram shows how many samples survive the intensity filter at each distance from the sonar,
with the range filter cutoffs drawn as dashed lines.

Arguments:
    <nc_file>: input NetCDF file with dragon aris3000 data
    <output_dir>: directory to which rendered frames will be written
    --start/--end: start and end frames for visualization
    --start-time/--end-time: start and end datetime timestamp for visualization (UTC), will approximate to closest timestamp
    --min-range/--max-range: specifies minimum and maximum extents for band-pass range filter
    --threshold: specifies high-pass intensity threshold value

Usage:
    python tools/aris_filter_staging.py <nc_file> <output_dir>
    python tools/aris_filter_staging.py <nc_file> <output_dir> --start 1000 --end 2000
    python tools/aris_filter_staging.py <nc_file> <output_dir> --start-time 2026-08-17T20:15:00 --end-time 2026-08-17T20:20:00
    python tools/aris_filter_staging.py <nc_file> <output_dir> --start 1000 --end 2000 --threshold 80
    python tools/aris_filter_staging.py <nc_file> <output_dir> --threshold 80 --min-range 2 --max-range 8
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
    parser.add_argument('--threshold', type=int, default=50, help='intensity a sample must exceed to survive the filter (default 50)')
    parser.add_argument('--min-range', type=float, default=0, help='closest range from the sonar (m) a sample must be at to survive the filter (default: no limit)')
    parser.add_argument('--max-range', type=float, default=float('inf'), help='farthest range from the sonar (m) a sample must be at to survive the filter (default: no limit)')
    parser.add_argument('--range-bins', type=int, default=100, help='number of bins in the range histogram (default 100)')
    args = parser.parse_args()
    if not 0 <= args.threshold < 255:
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

    # Fixed range bin edges covering every selected frame, so the range axis doesn't jump between recordings
    r_min, r_max = min_range[frames].min(), max_range[frames].max() # Get minimum and maximum range
    range_edges = np.linspace(r_min, r_max, args.range_bins + 1) # Get edges of these for visualization

    # Count every selected frame first, so both histograms' y-axes can be fixed to their tallest bar
    intensity_counts = np.zeros((len(frames), 256), dtype='i8') # Initialize intensity counter
    range_counts = np.zeros((len(frames), args.range_bins), dtype='i8') # Initialize range counter
    for k, i in enumerate(frames): # For each frame...
        image, ranges, in_range = read_frame(ds, i, args.min_range, args.max_range) # Read the frame
        intensity_counts[k] = np.bincount(image[in_range].ravel(), minlength=256) # Get every intensity value inside the range filter
        survivors_per_bin = (image > args.threshold).sum(axis=1) # Number of beams surviving the intensity filter at each bin
        range_counts[k] = np.histogram(ranges, bins=range_edges, weights=survivors_per_bin)[0] # Get every range value 
    print(f'counted {len(frames)} frames')

    # Colormap: at or below the threshold is black (under the colormap's range), survivors go blue to red,
    # samples outside the range filter (masked) are gray
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue_red', ['blue', 'red']).with_extremes(under='black', bad='gray') # Initialize color map
    norm = matplotlib.colors.Normalize(vmin=args.threshold + 0.5, vmax=255) # Map intensity values from 0 to 1

    # Generate figure: sonar fan on the left, intensity and range histograms stacked on the right
    fig, axes = plt.subplot_mosaic([['fan', 'intensity'], ['fan', 'range']], figsize=(13, 7), layout='constrained')
    ax_fan, ax_intensity, ax_range = axes['fan'], axes['intensity'], axes['range']

    # Sonar fan with one fixed extent covering every selected frame
    max_angle = np.deg2rad(np.abs(beam_angle).max()) # Determine extent of furthest angle
    ax_fan.set_xlim(-r_max * np.sin(max_angle), r_max * np.sin(max_angle)) # Set x-axis limits
    ax_fan.set_ylim(r_min * np.cos(max_angle), r_max) # Set y-axis limits
    ax_fan.set_aspect('equal') # Set equal aspect
    ax_fan.set_xlabel('Across-beam Distance (m)') # Set x-axis label
    ax_fan.set_ylabel('Range from Sonar (m)') # Set y-axis label
    ax_fan.set_facecolor('white') # Distinct from black (filtered out), the blue-red gradient (survivors) and gray (outside range filter)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=ax_fan, shrink=0.7, extend='min', label=f'Echo Intensity (Black: ≤ {args.threshold}, Filtered Out)') # Generate colorbar 
    colorbar.set_ticks([args.threshold + 1, 255]) # Sets numbers on colorbar

    # Intensity histogram, one bar per possible uint8 value, each colored the same as in the fan
    intensities = np.arange(256) # Initialize array
    intensity_bars = ax_intensity.bar(intensities, np.zeros(256), width=1, color=cmap(norm(intensities)), edgecolor='none') # Set color
    ax_intensity.set_xlim(-0.5, 255.5) # Set limit of x-axis based on possible intensity values with some headroom
    ax_intensity.set_ylim(0, max(intensity_counts.max(), 1) * 1.05) # Set limit to be tallest bar across the selected frames with some headroom
    ax_intensity.axvline(args.threshold, color='black', linestyle='--', label=f'Intensity Threshold ({args.threshold})') # Threshold line marker
    ax_intensity.set_xlabel('Echo Intensity (0 - 255)') # X-axis label
    ax_intensity.set_ylabel('Number of Samples') # Y-axis label
    ax_intensity.legend() # Visualize legend

    # Range histogram of samples surviving the intensity filter
    range_centers = (range_edges[1:] + range_edges[:-1]) / 2 # Get range centers again
    range_colors = np.where((range_centers >= args.min_range) & (range_centers <= args.max_range), 'red', 'gray') # Red inside the range filter, gray outside
    range_bars = ax_range.bar(range_centers, np.zeros(args.range_bins), width=np.diff(range_edges), color=range_colors, edgecolor='none') # Generate bar of number of cells with intensity over threshold at each range
    ax_range.set_xlim(r_min, r_max) # Set x-limit based on range minimum and maximum set by instrument during collection
    ax_range.set_ylim(0, max(range_counts.max(), 1) * 1.05) # Tallest bar across the selected frames, with a little headroom
    for cutoff in (args.min_range, args.max_range): # Draw cutoff lines for those given
        if r_min < cutoff < r_max: # Check that cutoffs are actually in range
            ax_range.axvline(cutoff, color='black', linestyle='--') # Draw to figure
    ax_range.set_xlabel('Range from Sonar (m)') # X-axis label
    ax_range.set_ylabel(f'Samples with Intensity > {args.threshold}') # Y-axis label

    mesh, window = None, None # Initialize figure
    for k, i in enumerate(frames): # For each frame...
        image, ranges, in_range = read_frame(ds, i, args.min_range, args.max_range) # Get frame, distance, and bins
        n_bins = in_range.size # Set size based on number of bins
        image = np.ma.masked_array(image, mask=np.broadcast_to(~in_range[:, np.newaxis], image.shape)) # Mask bins outside the range filter so they draw gray

        # Sonar fan
        frame_window = (min_range[i], max_range[i], n_bins) # Establish window from frame parameters
        if frame_window == window: # If geometry has been set by the previous frame...
            mesh.set_array(image) # Same fan geometry as the last frame, update pixels only
        else: # If geometry has not been set by the previous frame
            if mesh is not None: # And the mesh has been set...
                mesh.remove() # Remove the previous mesh
            x, y = fan_coordinates(min_range[i], max_range[i], n_bins, beam_angle) # Get cartesian coordinates for each cell from sonar polar
            mesh = ax_fan.pcolormesh(x, y, image, cmap=cmap, norm=norm, shading='flat') # Set mesh
            window = frame_window # Set window for next frame to reference geometry

        # Histograms
        for bar, count in zip(intensity_bars, intensity_counts[k]):
            bar.set_height(count) # Set height of each bar in intensity histogram
        for bar, count in zip(range_bars, range_counts[k]):
            bar.set_height(count) # Set height of each bar in range histogram
        fig.suptitle(
            f'frame {i} of {times.size - 1}    {str(times[i])[:23]}\n'
            f'{min_range[i]:.1f}-{max_range[i]:.1f} m, {n_bins} bins, {frame_rate[i]:.1f} FPS',
            fontsize=10,
        ) # Set title of whole figure
        fig.savefig(args.output_dir / f'frame_{i:0{digits}d}.png', dpi=120) # Save frame
        if (i - start) % 100 == 0:
            print(f'Frame {i}/{end}')

    plt.close(fig)
    print(f'Saved {len(frames)} frames to {args.output_dir}')

# =============================

if __name__ == '__main__':
    main()
