from StaticVariables import SCALE_FACTOR_EEG, gain_code, com_port, n_channels, daisy_used

import time
import numpy as np
from pyOpenBCI import OpenBCICyton
from pylsl import StreamInfo, StreamOutlet

# TO-DO: search for suspiciously large time gaps between samples
# TO-DO: cehck if any illegal trigger data is still being received

def Start():

    def shutdown_cyton_data_packager():
        print("Shutting down Cyton data packager...")

        ####################################
        #    Destroy UM232R FTDI object    #
        ####################################

        try:
            print("Destroying Cyton's OpenBCICyton object...")

            if cyton is not None:
                cyton.disconnect()
            else:
                print("'cyton' is not defined.")
        except NameError:
            print("'cyton' is not declared.")
        except Exception as e:
            raise e
        
        ####################################
        #        Destroy EEG outlet        #
        ####################################

        try:
            print("Destroying EEG outlet...")

            if stream_outlet is not None:
                # send shutdown signal to any consumers of this outlet
                data_packager_shutdown_signal = np.empty(n_channels+1) # ???

                # wait for consumers to disconnect
                while stream_outlet.have_consumers():
                    stream_outlet.push_sample(data_packager_shutdown_signal)

                    # wait for 100ms, then try again (instead of blocking)
                    # pause between calls allows for interrupts
                    time.sleep(0.1)

                # completely delete stimuli outlet
                stream_outlet.__del__()
            else:
                print("'stream_outlet' is not defined.")
        except NameError:
            print("'stream_outlet' is not declared.")
        except Exception as e:
            raise e
        
        # end of shutdown_cyton_data_packager
        print("Cyton data packager shut down.")

        try:
            while True:
                time.sleep(.1)
        except KeyboardInterrupt:
            pass
        pass

    def Run():
        # most recent EEG sample from Cyton
        current_sample_EEG = np.zeros((1,n_channels))

        # most recent trigger sample from Cyton
        current_sample_trigger = np.zeros(2)

        chunk_ind = 0
        chunk = np.zeros((500,n_channels+2))

        def package_sample(sample):
            nonlocal current_sample_EEG, current_sample_trigger, chunk_ind

            # pull sample from 8 EEG channels
            current_sample_EEG = np.float32(np.array(sample.channels_data)*SCALE_FACTOR_EEG)

            # pull sample from digital pins
            pins = sample.aux_data[0]

            # grab pin D11's inverted sample (MSB)
            current_sample_trigger[0] = pins < 256

            # grab pin D12's inverted sample (LSB)
            current_sample_trigger[1] = not(pins%2)

            # append formatted sample to chunk
            res = np.zeros(n_channels+2)
            res[0:n_channels] = current_sample_EEG
            res[n_channels:] = current_sample_trigger
            chunk[chunk_ind,:] = res

            # increment chunk index and see if it's time to send chunk
            chunk_ind += 1
            if chunk_ind == 500:
                stream_outlet.push_chunk(chunk.tolist())
                chunk_ind = 0
                print("chunk sent")

        # start board stream with callback
        cyton.start_stream(package_sample)

    cyton = None
    stream_outlet = None

    try:
        # connect to bluetooth dongle on COM port
        print("Connecting to Cyton...")
        cyton = OpenBCICyton(port=com_port, daisy=daisy_used)

        # configure board into digital read mode
        print("Configuring Cyton...")
        cyton.write_command('/d')
        cyton.write_command("x10" + str(gain_code) + "0110X")
        cyton.write_command("x20" + str(gain_code) + "0110X")
        cyton.write_command("x30" + str(gain_code) + "0110X")
        cyton.write_command("x40" + str(gain_code) + "0110X")
        cyton.write_command("x50" + str(gain_code) + "0110X")
        cyton.write_command("x60" + str(gain_code) + "0110X")
        cyton.write_command("x70" + str(gain_code) + "0110X")
        cyton.write_command("x80" + str(gain_code) + "0110X")
        cyton.write_command('/3')
        print("Cyton ready.")

        # initialize LSL stream
        print("Opening EEG outlet...")
        stream_info = StreamInfo("Packaged_EEG", "Packaged_EEG", n_channels+2, 250, "float32", "CytonDataPackager")
        stream_outlet = StreamOutlet(stream_info)
        print("EEG outlet opened.")

        Run()
    except KeyboardInterrupt:
        # check for something else here
        print("Keyboard interrupt detected")
        shutdown_cyton_data_packager()
    except Exception as e:
        shutdown_cyton_data_packager()
        raise e
    else:
        shutdown_cyton_data_packager()
    finally:
        import sys
        sys.exit(0)

Start()