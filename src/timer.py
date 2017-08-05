"""Track and predict how long an Amazon Cloud upload is going to take."""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md
import pytz
import os.path

from datetime import datetime
from scipy import stats
from ipdb import set_trace as st

class TimeIt(object):
    """Create a time-tracking object.

    Attributes:
        upload_data (list): Measurements of file number uploads.
        time_data (list): Measurements of the current system time when 
            each upload measurement was taken.
        nfiles (int): The total number of files to be uploaded.

    """

    def __init__(self, nfiles=100):
        """Return a TimeIt object.

        Note:
            Reads in measurements from local directory if previously
            recorded and continues; else starts with empty lists.

        Args:
            file_path (str): String of FITS file to load.

        """        
        self.upload_data = self.read_upload_data(init=1)
        self.time_data = self.read_time_data(init=1)
        self.nfiles = nfiles
        
        
    def add_to_dataset(self, file_number=None):
        """Record a measurement of current file being uploaded and a timestamp.

        Returns:
            True if successful, False otherwise.

        """   
        if file_number:
            current_file = file_number
        else:
            current_file = input("Current file # being uploaded: ")
        self.time_data.append(datetime.now())
        self.upload_data.append(self.nfiles - current_file)
        self.write_upload_data()
        self.write_time_data()
        return True
    
    def read_upload_data(self, init=0, file_path='data/uploads.txt'):
        """Return a list that holds the number of files being uploaded.

        Args:
            init (int): Flag for whether to start from scratch and ignore any
                saved previous measurements (set True if desired).
            file_path (str): Path for the uploads file.

        Returns:
            upload_data (list): List for holding upload data measurements.

        """
        if os.path.exists(file_path) and init == 0:
            return list(np.loadtxt(file_path, delimiter=','))
        else:
            return []
        
    def read_time_data(self, init=0, file_path='data/timestamps.txt'):
        """Return a list that holds the timestamps.

        Args:
            init (int): Flag for whether to start from scratch and ignore any
                saved previous measurements (set True if desired).
            file_path (str): Path for the timestamps file.

        Returns:
            time_data (list): List for holding upload data measurements.

        """ 
        if os.path.exists(file_path) and init == 0:
            timestamps = np.loadtxt(file_path, delimiter=',', dtype=str)
            time_data = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") for x in timestamps]
            return time_data
        else:
            return []
        
    def write_upload_data(self, file_path='data/uploads.txt'):
        """Write the upload data to a file.

        Args:
            file_path (str): Path for the uploads file.

        Returns:
            True if successful, False otherwise.

        """
        try:
            np.savetxt(file_path, self.upload_data, delimiter=',')
        except IOError as err:
            print("IO error: {0}".format(err))
            raise SystemExit(err)
        return
        
    def write_time_data(self, file_path='data/timestamps.txt'):
        """Write the timestamps to a file.

        Args:
            file_path (str): Path for the timestamps file.

        Returns:
            True if successful, False otherwise.

        """
        try:        
            np.savetxt(file_path, self.time_data, delimiter=',', fmt='%s')
        except IOError as err:
            print("IO error: {0}".format(err))
            raise SystemExit(err)
        return
    
    def time_data_to_dec(self, time_data):
        """Retreive the timestamps, including a decimal version for calculations.

        Returns:
            time_data_dec (numpy.ndarray): Floats of seconds since start of
                epoch.

        """        
        time_data_dec = np.array([(x-datetime(1970,1,1)).total_seconds() for x in time_data])
        return time_data_dec
    
    def plot_it(self, poly=2, **kwargs):
        """Plot data and estimated upload completion time.

        Args:
            poly (int): Degree of polynomial for fit.
            **kwargs: Keyword arguments for matplotlib.pyplot.axes.plot.

        Returns:
            True if successful, False otherwise.

        """
        time_data_dec = self.time_data_to_dec(self.time_data)

        # Plot data.
        fig, ax = plt.subplots(1, 1, figsize=(12,8), dpi=150)
        ax.plot(self.time_data, self.upload_data, 'o', **kwargs)
        fig.autofmt_xdate()
        ax.set_xlabel('Time', fontsize=16)
        ax.set_ylabel('Files remaining', fontsize=16)
        ax.tick_params(labelsize=14)
        
        # Fit data
        z = np.polyfit(time_data_dec, self.upload_data, poly)
        roots = np.roots(z)
        endtime_dec = roots[0]
        endtime = datetime.fromtimestamp(endtime_dec, pytz.utc)
        p = np.poly1d(z)
        xp = np.linspace(time_data_dec[0], endtime_dec, 100)
        newdates = [datetime.fromtimestamp(x, pytz.utc) for x in xp]  
        newfiles = p(xp)
        theargmin = np.nanargmin(np.abs(newfiles))
        completion_time = newdates[theargmin].strftime('%H:%M')
        ax.plot(newdates, newfiles, '--r', zorder=-1,
                label=str('Estimated completion time: ' + completion_time))
        ax.set_xlim(xmax=endtime)
        ax.set_ylim(ymin=0)
        
        # Clean up plot and save.
        xfmt = md.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(xfmt)
        ax.legend(loc=0, fontsize=16)
        # plt.show()
        save_name = 'completion_time_k' + str(poly) + '.pdf'
        fig.savefig(save_name,
                    format='pdf', bbox_inches='tight')
        print("Saved: ", save_name)

        return


if __name__ == '__main__':

    test_run = TimeIt(nfiles=100)
    test_run.add_to_dataset(file_number=10)
    test_run.add_to_dataset(file_number=20)
    test_run.add_to_dataset(file_number=28)
    test_run.add_to_dataset(file_number=37)
    test_run.add_to_dataset(file_number=47)
    test_run.plot_it(poly=1)



