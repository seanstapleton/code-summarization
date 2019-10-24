#!/usr/bin/python

import numpy
import argparse
import csv

headers = []

# num_lines of java snippets
num_lines = [14, 13, 15, 27, 23, 22, 18, 15, 18, 16, 13, 13, 16, 13, 19]

# helper function to convert column name to index once represented in
# numpy array
def tc ( mine ):
    global headers
    return headers.index( mine )


def constrain (array, type_column, type_value, constrained_column):
    ret = []
    for row in array:
        if row[ tc( type_column ) ] == type_value:
            ret.append( row[tc(constrained_column)] )

    return numpy.asarray(ret)
            

def normalize_times (array):
    ret = []
    
    
    for row in array:
        my_times = []
        for i in range(0, 15):
            my_times.append( row[ tc('{}-delta'.format(i)) ] )


        mean =  numpy.mean( numpy.array(my_times, dtype=numpy.float64) )
        std  =  numpy.std( numpy.array(my_times, dtype=numpy.float64) )
        ret.append( (mean, std)  )

    return ret
        


def main():
    global headers
    p = argparse.ArgumentParser()
    p.add_argument('--file', type=str, help='which file to process', default='pilot.csv')
    args = p.parse_args()

    with open (args.file) as f:
        data_iter = csv.reader( f )
        data = [data for data in data_iter]
    #ignore header row
    my_array = numpy.asarray( data[1:] )

    #so that the helper function can have access
    headers = data[0]

    # now, access data with my_array[ :, tc('name-of-column') ]
    # print( my_array[:,  [tc('1-answer'), tc('2-answer')]] )
    

    means = []

    for i in range(0, 15):
        means.append( numpy.mean( numpy.array(my_array[:, tc('{}-delta'.format(i) ) ], dtype=numpy.float64), dtype=numpy.float64 ) )

    human = []
    machine = []
    none = []
    for i in range(0, 15):
        my_human = constrain(my_array, '{}-type'.format(i), 'human', '{}-delta'.format(i))
        my_machi = constrain(my_array, '{}-type'.format(i), 'machine', '{}-delta'.format(i))
        my_none  = constrain(my_array, '{}-type'.format(i), 'none', '{}-delta'.format(i))

        human.append(numpy.array(my_human, dtype=numpy.float64))
        machine.append(numpy.array(my_machi, dtype=numpy.float64))
        none.append(numpy.array(my_none, dtype=numpy.float64))

    normalized_times = normalize_times(my_array)
    print(normalized_times)

    the_means = {}
    for i in range(0, 15):
        pass

    


        
    #print(numpy.corrcoef(means, num_lines))
    





if __name__ == '__main__':
    main()
