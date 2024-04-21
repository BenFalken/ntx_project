# Data Collection Software

`StaticVariables.py` - Static variables to configure streaming from the Cyton board (i.e COM port, sampling frequency, gain multiplier)
`CytonDataPackager.py` - Streams data from the Cyton board and packages it
`DataCollector.py` - Receives packages from `CytonDataPackager.py` and formats data into arrays

## Electrode Configuration

GND/REF
CH1 (gray) - C3
CH2 (purple) - Cz
CH3 (blue) - C4
CH4 (green) - P3
CH5 (yellow) - Pz
CH6 (orange) - P4
CH7 (red) - O1
CH8 (brown) - O2
