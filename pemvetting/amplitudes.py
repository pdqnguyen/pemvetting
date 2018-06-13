import numpy as np
import pandas as pd
import time
import os
import warnings
from utils import run_scans, read_omega_scan_summary

__coup_func_dir__ = '/home/philippe.nguyen/public_html/test/CoupFuncData'

def check_amplitudes(event, omega_dir, coup_func_dir=__coup_func_dir__, verbose=False):
    
    tt = time.time()    
    # Run strain omega scans and determine the lowest false rate that triggers the strain channel
    cw_dir = os.path.dirname(os.path.abspath(__file__))
    false_rates = ['1e-6', '1e-5', '1e-4', '1e-3', '5e-3']
    omega_strain_dir = os.path.join(omega_dir, '') + 'strain/'
    omega_full_dir = os.path.join(omega_dir, '') + 'full/'
    config_strain_template = os.path.join(cw_dir, 'config', 'config_strain_' + event.ifo + '.txt')
    config_full_template = os.path.join(cw_dir, 'config', 'config_full_' + event.ifo + '.txt')
    config_full_filename = os.path.join(omega_full_dir, 'config_full.txt')
    if not os.path.exists(omega_strain_dir):
        os.makedirs(omega_strain_dir)
    if not os.path.exists(omega_full_dir):
        os.makedirs(omega_full_dir)
    event.find_lowest_false_rate(false_rates, omega_strain_dir, config_strain_template)
    event.create_config(config_full_filename, config_full_template)
    full_scan_summary = run_scans(event.scan_time, config_full_filename, omega_full_dir)
#     out_dir = '/home/philippe.nguyen/public_html/PEMVettingResults/'
#     full_scan_summary = os.path.join(out_dir, event.graceid, event.ifo, 'omegascans', 'full', 'summary.txt')
    #### READ OMEGASCAN SUMMARY FILE ####
    omega_scan_df = read_omega_scan_summary(full_scan_summary)
    gw_exists = (omega_scan_df.loc[event.name].peakTime > 1)
    if gw_exists:
        strain_peak = omega_scan_df.loc[event.name].peakAmplitude
        darm_peak = strain_peak * 4e3
    else:
        darm_peak = None
    #### COUPLING FUNCTION CHECKS ####
    # Find omega scan channels that are PEM sensors
    name_search = r'(PEM.(CS|EX|EY).(ACC|MAG))'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        omega_scan_df = omega_scan_df[omega_scan_df.index.str.contains(name_search) & (omega_scan_df.peakAmplitude > 0)]
    channels = sorted(omega_scan_df.index.unique())
    coup_func_dict = {c: os.path.join(coup_func_dir, c.replace('_DQ', '') + '_composite_coupling_data.txt') for c in channels}
    failed = []
    results = {}
    for channel, cf_file in coup_func_dict.items():
        _, freq, ampl = omega_scan_df.loc[channel]
        # Load coupling function data
        try:
            cf = pd.read_csv(cf_file)
        except:
            if verbose:
                print('failed to load ' + cf_file)
            failed.append(channel)
            continue
        # Replace some text for output
        cf.flag.replace('Real','Measured', inplace=True)
        cf.flag.replace('Upper Limit','UpperLim', inplace=True)
        cf.flag.replace('Thresholds not met','WeakInjc', inplace=True)
        cf.flag.replace('No data','NoInjectn', inplace=True)
        # Extract coupling factor, flag, and DARM background for peak frequency
        try:
            factor, flag, darm_bg = nearest_factor(cf, freq)
        except:
            if verbose:
                print('failed to compute darm amplitude for ' + channel)
            failed.append(channel)
            continue
        # Estimate darm contribution
        darm_ampl = ampl * factor
        ratio_bg = darm_ampl / darm_bg
        if darm_peak is not None:
            ratio_peak = darm_ampl / darm_peak
        else:
            ratio_peak = 'NoGWsignal'
        results[channel] = [freq, ampl, factor, flag, darm_ampl, ratio_bg, ratio_peak]
    if verbose:
        print('coupling function vetting: {:.2e} seconds'.format(time.time() - tt))
    return results, failed

def nearest_factor(cf, freq):
    """
    Finds nearest coupling factor to a frequency, interpolating if there are none within 10 Hz.
    
    Parameters
    ----------
    cf : pandas.Dataframe object
        Coupling function data ('frequency', 'flag', 'factor_counts', 'darm')
    freq : float
        Frequency.
    
    Returns
    -------
    factor : float
        Nearest or interpolated coupling factor.
    flag : str
        Flag of factor.
    darm_bg : float
        Value of DARM background at freq
    """
    if np.all(cf.flag == 'NoInjectn'):
        return None, None, None
    cf_real = cf[cf.flag == 'Measured']
    cf_upper = cf[cf.flag == 'UpperLim']
    if cf_real.shape[0] > 0:
        # Index of nearest measured value
        idx_ = np.argmin(np.abs(cf_real.frequency - freq))
    elif cf_upper.shape[0] > 0:
        # Index of nearest upper limit
        idx_ = np.argmin(np.abs(cf_upper.frequency - freq))
    if (cf_real.shape[0] + cf_real.shape[0] > 0) and (np.abs(cf.loc[idx_, 'frequency'] - freq) < 10.):
        # If there are measured values or upper limits nearby, use them.
        factor = cf.loc[idx_, 'factor_counts']
        flag = cf.loc[idx_, 'flag']
        darm_bg = cf.loc[idx_, 'darm']
    else:
        # Interpolate from surrounding rows to get values at peak frequency
        cf_nonzero = cf[cf.flag != 'NoInjectn']
        if (freq > cf_nonzero.frequency.min()) and (freq < cf_nonzero.frequency.max()):
            idx1 = cf_nonzero[cf_nonzero.frequency <= freq].frequency.index[-1]
            idx2 = cf_nonzero[cf_nonzero.frequency >= freq].frequency.index[0]
            rows = cf_nonzero.loc[idx1:idx2]
            frequencies = np.asarray(rows.frequency)
            factors = np.asarray(rows.factor_counts)
            factor = np.interp(freq, frequencies, factors)
            darm_bgs = np.asarray(rows.darm)
            darm_bg = np.interp(freq, frequencies, darm_bgs)
            idx_nearest = np.argmin(np.abs(rows.frequency - freq))
            flag = cf_nonzero.loc[idx_nearest, 'flag']
        else:
            if freq < cf_nonzero.frequency.min():
                idx_ = cf_nonzero.index[0]
            else:
                idx_ = cf_nonzero.index[-1]
            factor = cf_nonzero.loc[idx_, 'factor_counts']
            flag = cf_nonzero.loc[idx_, 'flag']
            darm_bg = cf_nonzero.loc[idx_, 'darm']
    return factor, flag, darm_bg