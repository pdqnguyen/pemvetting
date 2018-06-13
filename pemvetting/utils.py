import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import os
import subprocess
import time
import warnings
from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram
from ligo.gracedb.rest import GraceDb, HTTPError

def get_event(graceid, ifos=['H1', 'L1']):
    """
    Get event from GraceDb.
    """
    client = GraceDb()
    event = client.event(graceid).json()
    event_dict = {}
    # Coincident detection attributes
    coinc_insp = event['extra_attributes']['CoincInspiral']
    instruments = event['instruments'].split(',')
    mchirp = coinc_insp['mchirp']
    coinc_end_time = coinc_insp['end_time'] + float(coinc_insp['end_time_ns'])*1e-9
    coinc_template_duration = estimate_duration(mchirp)
    coinc_start_time = coinc_end_time - coinc_template_duration
    coinc_dict = {
        'graceid': graceid,
        'mchirp': mchirp,
        'start_time': coinc_start_time,
        'end_time': coinc_end_time,
        'template_duration': coinc_template_duration
    }
    # Single detection attributes
    for i, ifo in enumerate(instruments):
        sngl_insp = event['extra_attributes']['SingleInspiral'][i]
        end_time = sngl_insp['end_time'] + float(sngl_insp['end_time_ns'])*1e-9
        start_time = end_time - sngl_insp['template_duration']
        sngl_dict = {
            'graceid': graceid,
            'mchirp': mchirp,
            'm1': sngl_insp['mass1'],
            'm2': sngl_insp['mass2'],
            's1z': sngl_insp['spin1z'],
            's2z': sngl_insp['spin2z'],
            'start_time': start_time,
            'end_time': end_time,
            'template_duration': sngl_insp['template_duration']
        }
        event_dict[ifo] = sngl_dict
    missing_ifos = sorted(set(ifos) - set(instruments))
    if len(missing_ifos) == len(ifos):
        # All ifos missing, use coinc attributes only
        for ifo in missing_ifos:
            event_dict[ifo] = coinc_dict.copy()
    elif len(missing_ifos) > 0 and len(missing_ifos) < len(ifos):
        # One but not all ifos are missing; use existing ifo attributes for the missing ones
        existing_ifo = list(set(instruments) - set(missing_ifos))[0]
        for ifo in missing_ifos:
            event_dict[ifo] = event_dict[existing_ifo].copy()
    return event_dict

def estimate_duration(mchirp, freq_threshold=30.):
    GM = 6.674e-11 * 1.989e30 * mchirp
    c = 2.998e8
    t_delta = (freq_threshold**(-8./3.)) * (GM/c**3)**(-5./3.) * (8*np.pi)**(-8./3.) * 5
    return t_delta

def run_scans(gpstime, config, out_dir, frame_path='/hdfs/frames/O2', cmap='parula'):
    """
    Run an omega scan via the wpipeline.
    
    Parameters
    ----------
    gpstime : float
        Scan time.
    config : str
        Configuration file.
    out_dir : str
        Output directory.
    """
    out_dir = os.path.join(out_dir, '')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if os.path.exists(out_dir + 'lock.txt'):
        subprocess.call('rm ' + out_dir + 'lock.txt', shell=True)
    cmd_split = [
        '/home/omega/opt/omega/bin/wpipeline scan',
        str(gpstime),
        '-c', config,
        '-f', frame_path,
        '-o', out_dir,
        '-m', cmap,
        '-r'
    ]
    cmd = ' '.join(cmd_split)
    print(cmd)
    subprocess.call(cmd, shell=True)
    summary_file = out_dir + 'summary.txt'
    return summary_file

def read_omega_scan_summary(filename):
    """
    Read an omega scan summary file into a pandas dataframe.
    """
    usecols = [0,1,2,5]
    names = ['channelName', 'peakTime', 'peakFrequency', 'peakAmplitude']
    oscan = pd.read_csv(filename, header=5, delim_whitespace=True, index_col=0, usecols=usecols, names=names)
    return oscan

def trim_foft(foft, threshold=100):
    """
    Trim frequency time-series to avoid issues with waveform blowing up at merger time.
    """
    df = np.abs(np.concatenate([[0.], np.diff(foft)]))
    idx = np.arange(len(foft))[df > threshold][0]
    return foft[:idx]

def get_spec(channel, t0, t1, qrange=(4,96), frange=(20,300), tres=0.001, fres=1, outseg=None, verbose=False):
    """
    Load data and create spectrogram from q-transform(s).
    
    Parameters
    ----------
    channels : list
    t0 : int, float
    t1 : int, float
    qrange : tuple, optional
    frange : tuple, optional
    tres : float, optional
    fres : float, optional
    outseg : tuple, optional
    
    Returns
    -------
    specs : list
    """
    if outseg is None:
        t = (t0 + t1) / 2.
        dt = 0.5
        outseg = (t - dt, t + dt)
    ts = TimeSeries.fetch(channel, t0, t1, verbose=verbose)
    spec = ts.q_transform(qrange=qrange, frange=frange, tres=tres, fres=fres, outseg=outseg)
    if spec.name is None:
        spec.name = channel
    return spec
    
def spec_plot(spec, file_name, epoch=None, tplot=None, fplot=None, verbose=False):
    """
    Plots a single spectrogram.
    
    Parameters
    ----------
    spec : gwpy.Spectrogram object
    file_name : str
    tplot : tuple, optional
    fplot : tuple, optional
    """
    plot = spec.plot(cmap=plt.get_cmap('viridis'))
    plt.title(spec.name.replace('_', ' '))
    if tplot is not None:
        plt.xlim(tplot)
    if fplot is not None:
        plt.ylim(fplot)
    plt.yscale('log')
    yticks = np.array([32, 64, 128, 256])
    plt.yticks(yticks, yticks.astype(str))
    if verbose:
        print(file_name)
    try:
        plot.savefig(file_name)
    except:
        print('failed to save ' + file_name)
    return plot

def vetting_results(filename, amplitude_results, contour_overlap_results, path_overlap_results, verbose=False):
    """
    Produces a table of the vetting results, and returns an overall 'pass'/'hin' state.
    
    Parameters
    ----------
    filename : str
        Output file (.txt)
    amplitude_results : dict
        Results of amplitude checks for each PEM channel.
    contour_overlap_results : dict
        Results of overlap with strain channel for each PEM channel.
    path_overlap_results : dict
        Results of overlap with event template for each PEM channel.
    verbose : bool
    
    Returns
    -------
    state : str
        Overall state for this interferometer 'pass' or 'hin' (human input needed).
    """
    
    tt = time.time()
    channels = sorted(amplitude_results.keys())
    final_results = {}
    with open(filename, 'wb') as file:
        # Create header
        header = '{:<40} {:<10} {:<10} {:<12} {:<10} {:<10} {:<9} {:<11} {:<12} {:<9} {:<5}'.format(
            'channel_name', 'peak_freq', 'peak_ampl', 'coup_factor', 'flag', 'darm_ampl', 'ratio_bg', 'ratio_peak', 'contour_ovr', 'path_ovr', 'state')
        file.write(header)
        for channel in channels:
            peak_freq, peak_ampl, coup_factor, flag, darm_ampl, ratio_bg, ratio_peak = amplitude_results[channel]
            if channel in list(contour_overlap_results.keys()):
                if contour_overlap_results[channel]:
                    contour_ovr = 'YES'
                else:
                    contour_ovr = 'NO'
            else:
                contour_ovr = 'NoChannel'
            if channel in list(path_overlap_results.keys()):
                if path_overlap_results[channel]:
                    path_ovr = 'YES'
                else:
                    path_ovr = 'NO'
            else:
                path_ovr = 'NoChannel'
            no_channel = ((contour_ovr == 'NoChannel') or (path_ovr == 'NoChannel'))
            overlapping = ((contour_ovr == 'YES') or (path_ovr == 'YES'))
            high_ratio = ((ratio_bg >= 0.1) or (ratio_peak >= 0.1))
            if no_channel:
                channel_state = 'HIN'
            elif overlapping and high_ratio:
                channel_state = 'HIN'
            else:
                channel_state = 'PASS'
            # Append data to TXT table
            line_data = [channel, peak_freq, peak_ampl, coup_factor, flag, darm_ampl,
                         ratio_bg, ratio_peak, contour_ovr, path_ovr, channel_state]
            if type(ratio_peak) == str:
                line_format = '{:<40} {:<10.2f} {:<10.2e} {:<12.2e} {:<10} {:<10.2e} {:<9.4f} {:<11} {:<12} {:<9} {:<5}'
            else:
                line_format = '{:<40} {:<10.2f} {:<10.2e} {:<12.2e} {:<10} {:<10.2e} {:<9.4f} {:<11.4f} {:<12} {:<9} {:<5}'
            line = ''
            for fmt, value in zip(line_format.split(' '), line_data):
                line += fmt.format(value) + ' '
            file.write('\n' + line)
            final_results[channel] = amplitude_results[channel] + [contour_ovr, path_ovr, channel_state]
    if len(final_results) > 0:
        # Save data to CSV
        cols = header.split()[1:]
#         cols = ['peak_freq', 'peak_ampl', 'coup_factor', 'flag', 'darm_ampl','ratio_bg', 'ratio_peak', 'contour_ovr', 'path_ovr', 'state']
        results_df = pd.DataFrame.from_dict(final_results, orient='index')
        results_df.index.rename('channel_name', inplace=True)
        results_df.columns = cols
        results_df.sort_index(inplace=True)
        results_df.to_csv(filename.replace('.txt', '.csv'))
        if np.any(results_df['state'] != 'PASS'):
            state = 'hin'
        else:
            state = 'pass'
    else:
        state = 'hin'
    if verbose:
        print('saved vetting results: {:.2e} seconds'.format(time.time() - tt))
    return state