# pemvetting
This code performs environmental checks for GW event candidates using PEM coupling functions and time-frequency signal overlaps. A "pass" state is reported only if all PEM channels have peak amplitudes below 10% of the strain channel peak amplitude, and if no PEM channels overlap the time-frequency path of the event; otherwise the state flag is "human input needed".

See http://pem.ligo.org/couplingfunctions/ for current PEM coupling function data and documentation.

More information on the PEM subsystem can be found at http://pem.ligo.org/channelinfo/.
