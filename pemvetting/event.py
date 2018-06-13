import numpy as np
import os
import subprocess
import warnings
import re
from copy import copy
from gwpy.detector import Channel
from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform, frequency_from_polarizations
from lalsimulation import SIM_INSPIRAL_FRAME_AXIS_ORBITAL_L
# Vetting code
import utils

class Event(Channel):
    """
    A single GW event candidate at a single interferometer.
    
    Attributes
    ----------
    name : str
        Name of GW strain channel.
    ifo : str
        Interferometer
    frametype : str
        Frame type for extracting data.
    sample_rate : int
        Sample rate (Hz).
    search_time_range : int
        Time range (seconds) for omega scan data extraction.
    frequency_range : tuple
        Min and max frequencies for omega scans.
    q_range : tuple
        Min and max Q values for omega scans.
    search_window_duration : int, float
        Time window (seconds) for omega scan search.
    plot_times : array-like
        Time windows (seconds) for omega scan plots.
    params : dict
        Dictionary of event parameters extracted from GraceDb. See utils.get_event for more info.
    tf_path : gwpy.timeseries.TimeSeries object
        Time-frequency path of candidate.
    """
    
    def __init__(self, graceid, ifo):
        self._init_event_attrs(graceid, ifo)
        self.tf_path = self._get_tf_path() # Time-frequency path (TimeSeries object)
        self._init_scan_attrs(ifo) # Omega scan parameters
        # Parameters to be used in overlap checks
        delta_t = 64
        self.t0 = self.scan_time - delta_t / 2.
        self.t1 = self.scan_time + delta_t / 2.
        self.outseg = (self.end_time - delta_t / 50. + 0.1,
                       self.end_time + 0.1)
        self.tres = delta_t / 1e5
        self.fres = np.diff(self.search_frequency_range)[0] / 300.
    
    def _init_event_attrs(self, graceid, ifo):
        """
        Initialize event attributes.
        """
        
        event_dict = utils.get_event(graceid)[ifo]
        for attr, value in event_dict.items():
            setattr(self, attr, value)

    def _init_scan_attrs(self, ifo):
        """
        Initialize attributes containing omega scan parameters.
        """
        if self.tf_path is None:
            search_window_duration = utils.estimate_duration(self.mchirp)
        else:
            search_window_duration = 1.25 * (self.end_time - self.tf_path.times.value.min())
        if search_window_duration > 10:
            search_time_range = 2048
            qrange = [75, 100]
            frange = [20, 2000]
        else:
            search_time_range = 64
            qrange = [4, 96]
            frange = [20, 300]
        plot_times = np.logspace(np.log10(search_window_duration), np.log10(self.template_duration), 4)[:-1]
        scan_params = {
            'frametype': '{}_HOFT_C00'.format(ifo),
            'sample_rate': 16384,
            'search_time_range': search_time_range,
            'search_frequency_range': frange,
            'search_q_range': qrange,
            'search_window_duration': search_window_duration,
            'plot_times': plot_times,
        }
        self.scan_time = 0.5 * (self.end_time + self.tf_path.times.value.min())
        super(Event, self).__init__(ifo + ':GDS-CALIB_STRAIN', **scan_params)
    
    def _get_tf_path(self, approximant='SEOBNRv4_opt', sample_rate=4096*4):
        """
        Get frequency track of an event based on GraceDb parameters.

        Parameters
        ----------
        approximant : str, optional
        f_lower : int, optional
        sample_rate : int, optional

        Returns
        -------
        tf_path : gwpy.timeseries.TimeSeries object
            Time series of frequency.
        """
        if not all(hasattr(self, x) for x in ['graceid', 'm1', 'm2', 's1z', 's2z', 'start_time', 'end_time']):
            return None
        hp, hc = get_td_waveform(
            approximant='SEOBNRv4_opt',
            mass1=self.m1,
            mass2=self.m2,
            spin1z=self.s1z,
            spin2z=self.s2z,
            delta_t=1.0/sample_rate,
            f_lower=30,
            frame_axis=SIM_INSPIRAL_FRAME_AXIS_ORBITAL_L
        )
        foft = frequency_from_polarizations(hp, hc)
        try:
            foft = utils.trim_foft(foft)
        except:
            pass
        tf_path = TimeSeries(foft, t0=self.end_time+float(foft.start_time), dt=foft.delta_t)
        tf_path = tf_path.crop(self.start_time, self.end_time)
        tf_path.name = self.graceid + '_path'
        return tf_path

    def find_lowest_false_rate(self, false_rates, omega_scan_dir, config_template):
        """
        Determine the lowest false alarm rate that triggers a strain omega scan.

        Parameters
        ----------
        false_rates : array-like
            False alarm rate values to run omega scans with.

        Returns
        -------
        output : str
            Lowest false rate that triggered a strain omega scan.
        """
        clist = OmegaScanConfig.from_file(config_template)
        sections = clist.sections
        strain_channel = sections['Calibrated h(t)'][0]
        clist_new = []
        for false_rate in false_rates:
            replace_mapping = {
                'searchTimeRange': self.search_time_range,
                'searchFrequencyRange': self.search_frequency_range,
                'searchQRange': self.search_q_range,
                'whiteNoiseFalseRate': false_rate,
                'searchWindowDuration': self.search_window_duration,
                'plotTimeRanges': list(self.plot_times)
            }
            new_channel = strain_channel.replace(replace_mapping)
            clist_new.append(new_channel)
        sections['Calibrated h(t)'] = clist_new
        # Save new config file
        filename = os.path.join(omega_scan_dir, '') + 'config_strain.txt'
        clist.save(filename)
        # Run strain omega scans
        omega_scan_dir = os.path.join(omega_scan_dir, '')
        if not os.path.exists(omega_scan_dir):
            os.makedirs(omega_scan_dir)
        strain_scan_summary = utils.run_scans(self.scan_time, filename, omega_scan_dir)  ### Run omega scans
        strain_scan = utils.read_omega_scan_summary(strain_scan_summary)
        # Determine lowest false alarm rate
        strain_scan['falseRate'] = false_rates
        strain_scan['falseRateFloat'] = strain_scan['falseRate'].astype(float)
        triggered_scans = strain_scan[strain_scan['peakTime'].astype(int) > 0]
        if triggered_scans.shape[0] > 0:
            lowest_false_rate = triggered_scans['falseRate'][triggered_scans['falseRateFloat'].values.argmin()]
            print('Lowest false alarm rate for ' + self.ifo + ': ' + lowest_false_rate)
        else:
            lowest_false_rate = '1e-3'
            print('Could not trigger CALIB_STRAIN omega scan. Choosing default false rate of 1e-3.')
        self.false_rate = lowest_false_rate
        return
    
    def create_config(self, config_filename, template_filename):
        """
        Create a configuration file for a full omega scan.

        Parameters
        ----------
        config_filename : str
            Configuration file name.
        template_filename : filename
            Template file for generating final config file.
        """
        clist = OmegaScanConfig.from_file(template_filename)
        replace_mapping = {
            'whiteNoiseFalseRate': self.false_rate,
            'searchWindowDuration': self.search_window_duration,
            'plotTimeRanges': list(self.plot_times)
        }
        for section in clist.sections.keys():
            section_new = []
            for channel in clist.sections[section]:
                if 'LOWFMIC' not in channel.channelName and 'SEIS' not in channel.channelName:
                    channel = channel.replace({'searchTimeRange': self.search_time_range})
                channel = channel.replace(replace_mapping)
                section_new.append(channel)
            clist.sections[section] = section_new
        clist.save(config_filename)
        return

class OmegaScanChannel(object):
    """
    Contains omega scan config options for a single channel.
    """
    
    def __init__(self, name, **params):
        self.channelName = name
        for key, value in params.items():
            setattr(self, key, value)
    
    @classmethod
    def from_string(cls, s):
        """
        Parse an omega scan entry for config options.
        
        Paramters
        ---------
        s : str
            Omega scan entry
        
        Returns
        -------
        new : OmegaScanChannel object
            Config options for a single channel.
        """
        
        patterns = [
            ('channelName', "'([A-Z0-9:\-_]+)'"),
            ('frameType', "'([A-Z0-9:\-_]+)'"),
            ('sampleFrequency', "([0-9]+)"),
            ('searchTimeRange', "([0-9]+)"),
            ('searchFrequencyRange', "\[([0-9]+\.*[0-9]*)?[ ]?([0-9]+)\]"),
            ('searchQRange', "\[([0-9]+)?[ ]?([0-9]+)\]"),
            ('searchMaximumEnergyLoss', "([0-9]+\.[0-9]+)"),
            ('whiteNoiseFalseRate', "([0-9]+e\-[0-9]+)"),
            ('searchWindowDuration', "([0-9]+\.*[0-9]*)"),
            ('plotTimeRanges', "\[([0-9]+\.*[0-9]*)?[ ]?([0-9]+\.*[0-9]*)?[ ]?([0-9]+\.*[0-9]*)?\]"),
            ('plotFrequencyRange', "\[([0-9]+\.*[0-9]*)?[ ]?([0-9]+\.*[0-9]*)?\]"),
            ('plotNormalizedEnergyRange', "\[([0-9]+\.*[0-9]*)?[ ]?([0-9]+\.*[0-9]*)?\]"),
            ('alwaysPlotFlag', "([0-9]?)")
        ]
        params = {}
        for attr, pattern in patterns:
            template = "[ ]+" + attr + ":[ ]+" + pattern
            match = re.search(template, s)
            if match is None:
                if attr in ['channelName', 'frameType', 'whiteNoiseFalseRate']:
                    value = ''
                else:
                    value = None
            elif len(match.groups()) > 1:
                value = list(match.groups())
            else:
                value = match.group(1)
            if type(value) == list:
                try:
                    value = [int(x) for x in value]
                except:
                    try:
                        value = [float(x) for x in value]
                    except:
                        pass
            else:
                try:
                    value = int(value)
                except:
                    if attr != 'whiteNoiseFalseRate':
                        try:
                            value = float(value)
                        except:
                            pass
            params[attr] = value
        name = params.pop('channelName')
        new = OmegaScanChannel(name, **params)
        return new
    
    def to_string(self):
        """
        Convert attributes to a formatted omega scan entry.
        
        Returns
        -------
        s : str
            Formatted string to be appended to an omega scan.
        """
        
        template = "{{\n" +\
        "  channelName:               '{channelName}'\n" +\
        "  frameType:                 '{frameType}'\n" +\
        "  sampleFrequency:           {sampleFrequency:.0f}\n" +\
        "  searchTimeRange:           {searchTimeRange:.0f}\n" +\
        "  searchFrequencyRange:      {searchFrequencyRange}\n" +\
        "  searchQRange:              {searchQRange}\n" +\
        "  searchMaximumEnergyLoss:   {searchMaximumEnergyLoss:.6f}\n" +\
        "  whiteNoiseFalseRate:       {whiteNoiseFalseRate}\n" +\
        "  searchWindowDuration:      {searchWindowDuration:.1f}\n" +\
        "  plotTimeRanges:            {plotTimeRanges}\n" +\
        "  plotFrequencyRange:        {plotFrequencyRange}\n" +\
        "  plotNormalizedEnergyRange: {plotNormalizedEnergyRange}\n" +\
        "  alwaysPlotFlag:            {alwaysPlotFlag}\n" +\
        "}}\n"
        replace_dict = self.__dict__
        for key, value in replace_dict.items():
            if type(value) == list and None in value:
                replace_dict[key] = []
            elif type(value) == list:
                value_str = '[' + ' '.join(['{:.1f}'.format(x) for x in value]) + ']'
                replace_dict[key] = value_str
            elif value is None:
                replace_dict[key] = ''
        s = template.format(**replace_dict)
        return s
    
    def replace(self, mapping):
        new = copy(self)
        for key, value in mapping.items():
            setattr(new, key, value)
        return new

class OmegaScanConfig(OmegaScanChannel):
    """
    Config parser for omega scans.
    """
    
    def __init__(self, filename, sections):
        self.filename = filename
        self.sections = sections
    
    @classmethod
    def from_file(cls, filename):
        """
        Parse omega scan file for config options.
        
        Parameters
        ----------
        filename : str
            Omega scan config file.
        
        Returns
        -------
        new : OmegaScanConfig object
            Contains config options separated by channel and section.
        """
        
        with open(filename) as f:
            lines = f.readlines()
        sections = {}
        i = 0
        while i < len(lines):
            if lines[i][0] == '[':
                sec = re.findall('\[(.*),.*\]', lines[i])[0]
                sections[sec] = []
            if lines[i].replace(' ','') == '{\n':
                j = i
                while j < len(lines):
                    if '}' in lines[j]:
                        entry = ''.join(lines[i:j+1])
                        sections[sec].append(entry)
                        break
                    else:
                        j += 1
                i = j
            else:
                i += 1
        out = {}
        for section, channels in sections.items():
            out[section] = [super(OmegaScanConfig, cls).from_string(channel) for channel in channels]
        new = OmegaScanConfig(filename, out)
        return new
    
    def save(self, file_out):
        """
        Save omega scan config options to a config file.
        
        Parameters
        ----------
        file_out : str
            output config filename.
        """
        
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i][0] == '[':
                break
            else:
                i += 1
        new = ''.join(lines[:i])
        section_names = sorted(self.sections.keys())
        if 'Context' in section_names:
            sec = 'Context'
            section_names.insert(0, section_names.pop(section_names.index(sec)))
        for sec in section_names:
            new += '\n[' + sec + ', ' + sec + ']\n'
            entries = self.sections[sec]
            for entry in entries:
                new += '\n' + entry.to_string()
        with open(file_out, 'w') as f:
            f.write(new)