import numpy as np
import time
from gwpy.spectrogram import Spectrogram
from scipy.ndimage.measurements import label
from utils import get_spec

def check_overlaps(event, channels, t_unc=0.001, f_unc=0, verbose=False):
    """
    Perform time-frequency overlap checks.
    
    Parameters
    ----------
    event : pemvetting.event.Event object
        Event object.
    channels : list
        Names of PEM channels to check.
    t_unc : float, int, optional
        Time uncertainty; widens path contour by this amount in time during path overlap check.
    f_unc : float, int, optional
        Frequency uncertainty; widens path contour by this amount in frequency during path overlap check.
    verbose : {False, True}
    
    Returns
    -------
    overlap_results : dict
        Contains list of spectrograms, dictionary of contour overlap results, and dictionary of path overlap results.
    failed : list
        Channels that failed to load.
    """
    
    tt = time.time()
    failed_channels = []
    sens_list = []
    for channel in channels:
        try:
            spec = get_spec(channel, event.t0, event.t1,
                        qrange=event.search_q_range,
                        frange=event.search_frequency_range,
                        tres=event.tres,
                        fres=event.fres,
                        outseg=event.outseg)
        except RuntimeError:
            if verbose:
                print('failed to load ' + channel)
            failed_channels.append(channel)
            continue
        sens_list.append(spec)
    if verbose:
        print('get sensor spectra: {:.2e} seconds'.format(time.time() - tt))
    missing_channels = [c for c in channels if c not in [sens.name for sens in sens_list]]
    contour_sens_list = [create_contour(sens) for sens in sens_list]
    if len(sens_list) > 0:
        times = sens_list[0].times.value
        freq_lengths = [len(sens.frequencies.value) for sens in sens_list]
        frequencies = sens_list[np.argmax(freq_lengths)].frequencies.value
        # WAVEFORM (PATH) OVERLAP
        if event.tf_path is not None:
            tt = time.time()
            tf_path = event.tf_path.crop(times[0], times[-1])
            contour_path = path_to_contour(tf_path, times, frequencies, t_unc=0.001, f_unc=0)
            if verbose:
                print('converting path to contour: {:.2e} seconds'.format(time.time()-tt))
            path_overlap_list = []
            path_overlaps = {}
            for contour_sens in contour_sens_list:
                tt1 = time.time()
                channel = contour_sens.name.replace('_contour', '')
                if len(contour_sens.frequencies.value) != len(contour_path.frequencies.value):
                    fmin = max([contour_path.frequencies.value.min(), contour_sens.frequencies.value.min()])
                    fmax = min([contour_path.frequencies.value.max(), contour_sens.frequencies.value.max()])
                    contour_path_cropped = contour_path.crop_frequencies(fmin, fmax)
                    contour_sens_cropped = contour_sens.crop_frequencies(fmin, fmax)
                    spec = contour_path_cropped * contour_sens_cropped
                else:
                    spec = contour_path * contour_sens
                overlap = spec.value.sum() > 0
                spec.name =  channel + '_path_overlap'
                path_overlap_list.append(spec)
                path_overlaps[channel] = overlap
        else:
            path_overlap_list = []
            path_overlaps = {c: True for c in channels}
        # SIGNAL (CONTOUR) OVERLAP
        if event.name is not None:
            tt = time.time()
            spec_gw = get_spec(event.name, event.t0, event.t1,
                            qrange=event.search_q_range,
                            frange=event.search_frequency_range,
                            tres=event.tres,
                            fres=event.fres,
                            outseg=event.outseg)
            if verbose:
                print('get gw spectrogram: {:.2e} seconds'.format(time.time()-tt))
            contour_gw = create_contour(spec_gw, largest=True)
            contour_overlap_list = []
            contour_overlaps = {}
            for contour_sens in contour_sens_list:
                channel = contour_sens.name.replace('_contour', '')
                if len(contour_gw.frequencies.value) != len(contour_sens.frequencies.value):
                    fmin = max([contour_gw.frequencies.value.min(), contour_sens.frequencies.value.min()])
                    fmax = min([contour_gw.frequencies.value.max(), contour_sens.frequencies.value.max()])
                    contour_gw_cropped = contour_gw.crop_frequencies(fmin, fmax)
                    contour_sens_cropped = contour_sens.crop_frequencies(fmin, fmax)
                    spec = contour_gw_cropped * contour_sens_cropped
                else:
                    spec = contour_gw * contour_sens
                overlap = spec.value.sum() > 0
                spec.name = channel + '_contour_overlap'
                contour_overlap_list.append(spec)
                contour_overlaps[channel] = overlap
            specs = [spec_gw, contour_gw, contour_path] + sens_list + contour_sens_list + contour_overlap_list + path_overlap_list
        else:
            contour_overlaps = {c: True for c in channels}
            specs = [contour_path] + sens_list + contour_sens_list + path_ovrlap_list
    else:
        # No channel data, just get strain if possible
        contour_overlaps = {}
        path_overlaps = {}
        if event.name is not None:
            spec_gw = get_spec(event.name, event.t0, event.t1,
                            qrange=event.search_q_range,
                            frange=event.search_frequency_range,
                            tres=event.tres,
                            fres=event.fres,
                            outseg=event.outseg)
            contour_gw = create_contour(spec_gw, largest=True)
            specs = [spec_gw, contour_gw]
        else:
            specs = []
    overlap_results = {'spectrograms': specs,
                       'contour_overlap': contour_overlaps,
                       'path_overlap': path_overlaps,
                       'failed_channels': failed_channels}
    return overlap_results

def create_contour(spec, largest=False):
    """
    Create a mask spectrogram, whose values are 1 where the input spectrogram is above background, and 0 otherwise.
    
    Parameters
    ----------
    spec : gwpy.Spectrogram object
        Original spectrogram.
    largest : {True, False}, optional
        If True, keep only the largest cluster of pixels in mask_spec.
    
    Returns
    -------
    contour_spec : gwpy.Spectrogram object
        Mask spectrogram.
    """
    contour_spec = spec.copy()
    contour_spec.name = spec.name + '_contour'
    if 'STRAIN' in contour_spec.name:
        thresh = 1
    else:
        thresh = 4
    mask = contour_spec.value > (contour_spec.value.mean() + thresh * contour_spec.value.std())
    contour_spec.value[:,:] = mask * 1
    if largest:
        # Keep only largest mask (i.e. largest cluster of pixels)
        labels,_ = label(contour_spec.value)             # Label clusters
        label_counts = np.bincount(labels.flatten())[1:] # Sizes of clusters
        largest_cluster = np.argmax(label_counts) + 1    # Label of largest cluster
        contour_spec.value[:,:] = (labels == largest_cluster) * 1
    return contour_spec

def path_to_contour(tf_path, times, frequencies, t_unc=0, f_unc=0):
    """
    Convert a time-frequency path as a TimeSeries object to a Spectrogram.
    
    Parameters
    ----------
    tf_path : gwpy.TimeSeries object
        Time-frequency path.
    times : array
        Time bins of output spectrogram
    frequencies : array
        Frequency bins of output spectrogram
    t_unc : float, int, optional
        Time uncertainty; widens path contour by this amount in time.
    f_unc : float, int, optional
        Frequency uncertainty; widens path contour by this amount in frequency.
    
    Returns
    -------
    contour_path : gwpy.Spectrogram object
        Spectrogram whose values are 1 in pixels overlapping the time-frequency path and 0 elsewhere.
    """
    path_times = tf_path.times.value
    path_frequencies = tf_path.value
    values = np.zeros([times.size, frequencies.size])
    for i in range(times.size-1):
        for j in range(frequencies.size-1):
            t_idx = np.where((path_times >= times[i] - t_unc ) & (path_times < times[i+1] + t_unc))[0]
            f_idx = np.where((path_frequencies[t_idx] >= frequencies[j] - f_unc) & (path_frequencies[t_idx] < frequencies[j+1] + f_unc))[0]
            if len(f_idx) > 0:
                values[i, j] = 1
    contour_path = Spectrogram(values, times=times, frequencies=frequencies)
    contour_path.name = tf_path.name
    return contour_path