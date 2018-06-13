import numpy as np
import pandas as pd
import os
from gwpy.detector import Channel
# PEM vetting modules
from utils import vetting_results, spec_plot
from event import Event
from amplitudes import check_amplitudes
from overlaps import check_overlaps

# Default output directory
__out_dir__ = '/home/philippe.nguyen/public_html/PEMVettingResults'

def vetting(graceid, verbose=False):
    """
    Perform PEM vetting of GW event candidate from GraceDb id number.
    
    Parameters
    ----------
    graceid : str
        GraceDb id for the event.
    
    Returns
    -------
    state : str
        'pass' or 'hin' for human input needed.
    """

    cw_dir = os.path.dirname(os.path.abspath(__file__))
    # False alarm rates to test strain channel omega scans
    false_rates = ['1e-6', '1e-5', '1e-4', '1e-3', '5e-3']
    # 'pass'/'hin' state for each interferometer
    all_states = []
    for ifo in ['H1', 'L1']:
        results_dir = __out_dir__ + '/' + str(graceid) + '/' + ifo
        results_filename = os.path.join(results_dir , 'vetting_results.txt')
        failed_channels_filename = os.path.join(results_dir, 'failed_channels.txt')
        plots_dir = os.path.join(results_dir, 'vettingplots')
        # Create Event object from GraceDb info
        event = Event(graceid, ifo)        
        #### COUPLING FUNCTION CHECKS ####
        amplitude_results, failed = check_amplitudes(event, results_dir + '/omegascans', verbose=verbose)
        if verbose:
            print('PEM amplitude checks for ' + ifo + ' complete.')
        channels = sorted(amplitude_results.keys())
        failed_dict = {channel: 'NoCouplingFunctionFound' for channel in failed}
        for key, value in failed_dict.items():
            print(key + ' - ' + value)
        #### SIGNAL OVERLAP CHECKS ####
        overlap_results = check_overlaps(event, channels, verbose=verbose)
        if verbose:
            print('Time-frequency overlap checks for ' + ifo + ' complete.')
        contour_overlap_results = overlap_results['contour_overlap']
        path_overlap_results = overlap_results['path_overlap']
        specs = overlap_results['spectrograms']
        for channel in overlap_results['failed_channels']:
            failed_dict[channel] = 'ChannelFailedToLoad'
        #### SAVE RESULTS ####
        # Report channels with no composite coupling functions found
        with open(failed_channels_filename, 'wb') as file:
            failed_channels = sorted(failed_dict.keys())
            lines = []
            for channel in failed_channels:
                lines.append('{:<45}{}'.format(channel, failed_dict[channel]))
            file.write('\n'.join(lines))
            if len(failed_channels) > 0 and verbose:
                print('Failed channels:')
                for c in failed_channels:
                    print(c)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        # State: 'pass', 'fail', or 'hin' (human input needed)
        single_state = vetting_results(results_filename, amplitude_results, contour_overlap_results, path_overlap_results, verbose=verbose)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
#         for spec in specs:
#             spec_plot(spec, plots_dir + '/{}.png'.format(spec.name.replace('_DQ','')), verbose=verbose)
        outseg_t = (event.outseg[1] - event.outseg[0])
        plot_delta_t = min([event.plot_times[0], outseg_t / 2.])
#         tplot = (event.scan_time - 0.6 * plot_delta_t,
#                  event.scan_time + 0.6 * plot_delta_t)
        tplot = (event.end_time - plot_delta_t + 0.1, event.end_time + 0.1)
        for spec in specs:
            spec_plot(spec, plots_dir + '/{}.png'.format(spec.name.replace('_DQ','')), tplot=tplot, verbose=verbose)
        all_states.append(single_state)
    if ('hin' in all_states):
        state = 'hin'
    else:
        state = 'pass'
    return state