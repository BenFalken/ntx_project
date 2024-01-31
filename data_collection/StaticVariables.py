from datetime import datetime

# hardware parameters
com_port = 'COM6'

sampling_frequency = 250

gain_code = 6

gain_multiplier = {
    0: 1,
    1: 2,
    2: 4,
    3: 6,
    4: 8,
    5: 12,
    6: 24
}

#gain_multiplier = 12
SCALE_FACTOR_EEG = (4500000) / gain_multiplier[gain_code] / (2**23-1)

# dynamic file naming
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d %H%M%S")

data_file_name = dt_string + ".csv"