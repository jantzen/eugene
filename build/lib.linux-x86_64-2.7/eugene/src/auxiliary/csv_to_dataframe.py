# csv_to_dataframe.py

""" Converts one or more csv files into a single DataFrame object. It is assumed
the files are stored with samples in rows. The first column corresponds to the
index variables. There is one additional column for each target variable. If
more than one file is provided as input (in a list of files), each is considered
a separate measurement of the same n target variables for the same system. All
files must be of the same dimension (same number of rows and columns). The first
columns of every file must be identical (identical index values).
"""

import eugene as eu
import csv
import numpy as np

def CSVtoDataFrame(files, index_id=0):

    # Open the first file and determine necessary parameters
    with open(files[0], 'rb') as csvfile:
        filereader = csv.reader(csvfile)
        temp = filereader.next()

    num_target_vars = len(temp) - 1
    target_ids = range(1,num_target_vars+1)

    # Create the empty DataFrame
    data = eu.interface.DataFrame(index_id, target_ids=target_ids)
    index_vals = []
    target_vals =[]
    
    # Loop over files
    for filename in files:
        with open(filename, 'rb') as csvfile:
            filereader = csv.reader(csvfile)
            index_temp = []
            target_temp =[]
            for row in filereader:
                index_temp.append(row[0])
                target_temp.append(row[1:])

            if len(index_vals) == 0:
                index_vals = np.array(index_temp,dtype="float64").reshape(len(index_temp),1)
#            else:
#                # Check whether the index values are actually the same
#                try:
#                    current_index_vals = np.array(index_temp).reshape(len(index_temp),1)
#                    assert np.array_equal(index_vals, current_index_vals)
#                except:
#                    raise ValueError("Index values not the same across all files.")
            
            target_vals.append(np.array(target_temp,dtype="float64"))

    # Fill in the data frame
    data._index_values = index_vals
    data._target_values = target_vals

    return data
