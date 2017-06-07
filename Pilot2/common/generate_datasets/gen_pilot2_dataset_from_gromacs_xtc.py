#!bin/python

"""
	Program to extract features GROMACS-generated
	MD trajectory files and save them to numpy
	binary file format, to be used as the ECP CANDLE
	project benchmarks.

	Author: Alfredo Metere (metere1@llnl.gov)
	Date: 05/30/2017 (MM/DD/YYYY)

	Copyright 2017 Lawrence Livermore National Laboratory.
"""

import numpy as np
from scipy.spatial import SphericalVoronoi

# Change according to your specific installation path of mdreader
import MDreader.mdreader as mdreader
#~ import mdreader

import MDAnalysis
import sys
from math import *
from time import *

import multiprocessing as mp
from multiprocessing import Process, Queue

def voro3D(points=np.array([0, 0, 0]), center=np.array([0, 0, 0])):
    #~ center = np.array([np.mean(points), np.mean(points), np.mean(points)])

    radius = np.amax(points)
    #~ print "Center = %25.20f\t%25.20f\t%25.20f" % (center[0],center[1], center[2])
    #~ print "Radius = %25.20f" % radius
    #~ print points.shape
    # calculate spherical Voronoi diagram
    sv = SphericalVoronoi(points, radius, center)
    # sort vertices (optional, helpful for plotting)
    sv.sort_vertices_of_regions()
    return sv

# This routine generates a onehot encoding vector for c=3 lipid types
def onehot(x, c=3):
    y = np.zeros(c)
    if x > c - 1:
        print 'error, label greater than number of classes'
        sys.exit(0)
    y[x] = 1
    return y

# This dictionaries are used for the onehot encoders
typeDict = {'CHOL': 0, 'DPPC': 1, 'DPPX': 1,
            'DIPC': 2, 'DIPX': 2, 'DOPC': 2, 'DOPX': 2}
locDict = {'GL1': 0, 'GL2': 0, 'NC3': 0, 'PO4': 0, 'ROH': 0,
           'C1A': 1, 'C1B': 1, 'C2A': 1, 'C2B': 1, 'C3A': 1, 'C3B': 1, 'C4A': 1, 'C4B': 1,
           'D2A': 1, 'D2B': 1, 'D3A': 1, 'D3B': 1,
           'R1': 1, 'R2': 1, 'R3': 1, 'R4': 1, 'R5': 1,
           'C1': 1, 'C2': 1, 'C3': 1}

# Uncomment the following two lines to have mdreader
# override the command-line arguments. Change the file paths
# accordingly
#~ mdsys = mdreader.MDreader(internal_argparse=False)

#~ mdsys.setargs(s="lKRAS/run1/10us.35fs-DPPC.0-DOPC.50-CHOL.50.tpr",
 #~ f="lKRAS/run1/10us.35fs-DPPC.0-DOPC.50-CHOL.50.xtc",
 #~ o="run1_10us.35fs-DPPC.0-DOPC.50-CHOL.50.npy")

#mdsys.setargs(s="10us.35fs-DPPC.0-DOPC.50-CHOL.50.tpr",
# f="10us.35fs-DPPC.0-DOPC.50-CHOL.50.xtc",
# o="run1_10us.35fs-DPPC.0-DOPC.50-CHOL.50.npy")

# Crystal
# mdsys.setargs(s="/g/g92/metere1/1TB/KRAS/run51/10us.35fs-DPPC.100-DOPC.0-CHOL.0.tpr",
# f="/g/g92/metere1/1TB/KRAS/run51/10us.35fs-DPPC.100-DOPC.0-CHOL.0.xtc",
# o="run51_10us.35fs-DPPC.100-DOPC.0-CHOL.0.npy")

# If the previous two lines are uncommented, you'll
# have to comment the following line:

def read_mddata():
    """ Read the xtc and trajectory file using MDreader
    """
    mdsys = mdreader.MDreader()
    mdsys.do_parse()
    print "Done"
    return mdsys

def select_trajectory(mdsys):
    """ Extract the trajectory from the simulation data
    """
    mdt = mdsys.trajectory
    return mdt

def select_atoms(mdsys):
    """ Return the selection of atoms that are of interest in the simulation
    """
    # Change this selection in case you have different lipids, and update the
    # onehot encoder dictionary accordingly
    asel = mdsys.select_atoms("resname DPPC or resname DOPC or resname DOPX or resname DPPX or resname CHOL or resname DIPC or resname DIPX")
    #~ asel_tails = mdsys.select_atoms("(resname DPPC or resname DOPC or resname DOPX or resname DPPX or resname CHOL or resname DIPC or resname DIPX) and not (name GL1 or name GL2 or name NC3 or name PO4 or name ROH)")
    #~ asel_heads = mdsys.select_atoms("(resname DPPC or resname DOPC or resname DOPX or resname DPPX or resname CHOL or resname DIPC or resname DIPX) and (name GL1 or name GL2 or name NC3 or name PO4 or name ROH)")

    return asel

def partition_work(mdsys):
    # Total number of frames in the MD trajectory
    totframes = len(mdsys)

    # Number of frames saved in each numpy array file
    fchunksize = 100

    totalchunks = 0
    totcheck = float(totframes) / float(fchunksize)
    totcheckint = int(totcheck)

    if (float(totcheck) - float(totcheckint) > 0):
        totalchunks = totcheckint + 1

    print "Total frames: %d, frames per chunk: %d, chunks: %d" % (totframes, fchunksize, totalchunks)
  
    return (totframes, fchunksize, totalchunks)

def process_gromacs_xtc(queue, processname, totframes, fchunksize, totalchunks, first_frame, last_frame, starting_chunk):
    mdsys = read_mddata()
    mdt = select_trajectory(mdsys)
    asel = select_atoms(mdsys)
    frags = asel.residues

    print "%d: Processing frames: %d - %d" % (processname, first_frame, last_frame)

    outA = np.zeros([fchunksize, len(frags)])
    outA.shape

    i = 0
    chunkcount = starting_chunk + 1 # Offset the chunkcount by 1
    lastchunksize = totframes - (totalchunks * fchunksize)
    outAL = []
    for curframe in range(first_frame, last_frame):
        j = last_frame - curframe
        mdt[curframe]
        print "[%d] Processing frame %d, %d remaining.\r" % (processname, mdt[curframe].frame, j)
        outL = []
        for curfrag in range(len(frags)):
            fr = frags[curfrag]
            ffr = np.zeros([12, 20])
            if (len(fr) < 12):
                ffr[:fr.positions.shape[0], :fr.positions.shape[1]] = fr.positions
                voro = voro3D(ffr[0:8, 0:3], fr.center_of_mass())
            else:
                ffr[:, 0:3] = np.array([fr.positions])
                voro = voro3D(ffr[:, 0:3], fr.center_of_mass())

            ohenc = np.zeros([3])
            ohenc = map(lambda x: x, onehot(typeDict[fr.residues.resnames[0]]))
            ffr[:, 3:6] = ohenc

            lipid_beads = fr.names

            for curatom in range(len(fr)):
                bmatrix = np.zeros([12])
                bead = fr.atoms[curatom]
                ohenc2 = np.zeros([2])
                ohenc2 = map(lambda x: x, onehot(locDict[bead.name], 2))
                ffr[curatom, 6:8] = ohenc2
                curbond = bead.bonds
                count = 0
                for ib in range(len(curbond)):
                    curblist = curbond.bondlist[ib]
                    curbead = curblist.atoms[0].name
                    #~ print "beads in bond: ", curblist.atoms[0].name, curblist.atoms[1].name
                    if curbead != bead.name:
                        bmatrix[lipid_beads == curbead] = curblist.length()
                    else:
                        bmatrix[lipid_beads == curblist.atoms[1].name] = curblist.length()

                #~ ffr[curatom, 8:] = bmatrix
                ffr[curatom, 8:] = bmatrix
                #~ print lipid_beads
                #~ print "Curr bead: ", bmatrix

            outL.append([ffr,voro.vertices])

        outLn = np.array(outL)
        outAL.append([outLn])
        outA = np.array(outAL)
        # In case of debug, uncomment the line below
        #~ print "outA.shape = %s" % str(outA.shape)

        # Flush the frames to disk
        if (i == fchunksize - 1):
            flush_chunk_to_file(processname, i, outA, mdsys.opts.outfile, chunkcount, totalchunks)
            i = -1
            outAL = []  
            chunkcount = chunkcount + 1

        i = i + 1

    if (i != 0):
        # Saves to disk the eventually remaining frames after
        # the last full chunk of data has been written
        flush_chunk_to_file(processname, i, outA, mdsys.opts.outfile, chunkcount, totalchunks)

def flush_chunk_to_file(processname, i, outA, outfile, chunkcount, totalchunks):
    myfilename = outfile + str(chunkcount).zfill(2) + \
                 "_outof_" + str(totalchunks) + ".npy"
    print "[%d] Flushing chunk (%d records) %d out of %d to file %s" % (processname, i + 1, chunkcount, totalchunks, myfilename)
    #~ np.save(myfilename, convert_to_helgi_format(outA))
    np.save(myfilename, outA)

def main():
    mdsys = read_mddata()
    (totframes, fchunksize, totalchunks) = partition_work(mdsys)

    """Create one process per cpu core and process multiple chunks in parallel
    """
    n = min(mp.cpu_count(), totalchunks)
    queues = []
    processes = []
    chunks_per_task = int(ceil(float(totalchunks) / float(n)))
    print "Using %d ranks and break up the work into %d chunks per task" % (n, chunks_per_task)
    starting_frame = 0
    for i in range(0,n):
        queues.append(Queue())
        ending_frame = min((i+1) * (chunks_per_task * fchunksize), totframes)
        starting_chunk = i * chunks_per_task
        process = Process(target=process_gromacs_xtc, args=(queues[i], i, totframes, fchunksize, totalchunks, starting_frame, ending_frame, starting_chunk))
        processes.append(process)
        starting_frame = ending_frame
        print "Starting process %d" %(i)

    for i in range(0,n):
        processes[i].start()

    for i in range(0,n):
        processes[i].join()

if __name__ == '__main__':
    main()
