#!bin/python

"""
    Program to extract features GROMACS-generated
    MD trajectory files and save them to numpy
    binary file format, to be used as the ECP CANDLE
    project benchmarks.

    Authors: Alfredo Metere (metere1@llnl.gov),
             Piyush Karande (karande1@llnl.gov),
             Brian Van Essen (vanessen1@llnl.gov)
    Date: 06/06/2017 (MM/DD/YYYY)

    Copyright 2017 Lawrence Livermore National Laboratory.
"""
import warnings

import numpy as np
# from scipy.spatial import SphericalVoronoi

# Change according to your specific installation path of mdreader
# import MDreader.mdreader as mdreader
import mdreader

import MDAnalysis
from MDAnalysis.analysis import distances, leaflet
import sys
from math import *
# import time

import multiprocessing as mp
from multiprocessing import Process, Queue

from neighborhood import Neighborhood

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=DeprecationWarning)


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
# mdsys = mdreader.MDreader(internal_argparse=False)

# mdsys.setargs(s="lKRAS/run1/10us.35fs-DPPC.0-DOPC.50-CHOL.50.tpr",
# f="lKRAS/run1/10us.35fs-DPPC.0-DOPC.50-CHOL.50.xtc",
# o="run1_10us.35fs-DPPC.0-DOPC.50-CHOL.50.npy")

# mdsys.setargs(s="10us.35fs-DPPC.0-DOPC.50-CHOL.50.tpr",
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
    asel = mdsys.select_atoms("resname DPPC or resname DOPC or resname DOPX or \
                               resname DPPX or resname CHOL or resname DIPC or resname DIPX")
    # asel_tails = mdsys.select_atoms("(resname DPPC or resname DOPC or resname DOPX or resname DPPX or resname CHOL or resname DIPC or resname DIPX) and not (name GL1 or name GL2 or name NC3 or name PO4 or name ROH)")
    # asel_heads = mdsys.select_atoms("(resname DPPC or resname DOPC or resname DOPX or resname DPPX or resname CHOL or resname DIPC or resname DIPX) and (name GL1 or name GL2 or name NC3 or name PO4 or name ROH)")

    return asel


def find_leaflets(mdsys):
    """ Find and leaflets in the simulation using PO4 beads
    """
    # print "Current frame: ", mdsys.trajectory.frame
    cut = 15
    while 1:
        lfls = leaflet.LeafletFinder(mdsys, 'name PO4', cutoff=cut, pbc=True)
        if len(lfls.groups()) <= 2:
            break
        else:
            cut += 0.5

    if len(lfls.groups()) != 2:
        raise ValueError("Problem don't have x2 leaflets.")

    top_head, bot_head = lfls.groups()
    rt = float(len(top_head))/len(bot_head)
    if rt > 1.3 or rt < 0.77:
        raise ValueError("Uneven leaflets.")

    return top_head, bot_head


def get_neighborhood(leaflet, mdsys):

    """ Get neighborhood object for the give leaflet
    """
    dist = distances.distance_array(leaflet.positions, leaflet.positions, mdsys.dimensions[:3])
    nbrs = Neighborhood(leaflet.positions, dist, mdsys.dimensions[:3])

    return nbrs


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


def process_gromacs_xtc(queue, processname, totframes, fchunksize, totalchunks,
                        first_frame, last_frame, starting_chunk,
                        mdsys, top_head, bot_head):

    mdt = select_trajectory(mdsys)
    asel = select_atoms(mdsys)
    frags = asel.residues

    print "%d: Processing frames: %d - %d" % (processname, first_frame, last_frame)

    # Generating names for the structured data
    names = ['x', 'y', 'z', 'CHOL', 'DPPC', 'DIPC', 'Head', 'Tail']
    for i in range(12):
        temp = 'BL'+str(i+1)
        names.append(temp)

    print "Feature names:\n", names

#    outA = np.zeros([fchunksize, len(frags), 12], dtype={'names':names, 'formats':['float']*len(names)})
    outA = np.zeros([fchunksize, len(frags), 12, 20])
    outNbrs = -1*np.ones([fchunksize, len(frags), 100])

    outResnums = frags.resnums
    # print outResnums
    # print outResnums.shape
    resnum_dict = dict(zip(outResnums, range(len(outResnums))))
    # outA.shape

    i = 0
    chunkcount = starting_chunk + 1  # Offset the chunkcount by 1
    lastchunksize = totframes - (totalchunks * fchunksize)
    # last_frame = first_frame + 300
    for curframe in range(first_frame, last_frame):
        j = last_frame - curframe
        # frame_ind = curframe - first_frame
        mdt[curframe]

        # Expand leaflets by adding CHOL molecules
        chol_head = mdsys.select_atoms("resname CHOL and name ROH")

        tp = top_head + chol_head.select_atoms("around 15 global group topsel", topsel=top_head)
        bt = bot_head + chol_head.select_atoms("around 15 global group botsel", botsel=bot_head)

        # Get neighborhood
        top_nbrs = get_neighborhood(tp, mdsys)
        bot_nbrs = get_neighborhood(bt, mdsys)

        print "[%d] Processing frame %d, %d remaining.\r" % (processname, mdt[curframe].frame, j)

        for curfrag in range(len(frags)):

            fr = frags[curfrag]
            ffr = np.zeros([12, 20])
            if (len(fr) < 12):
                ffr[:fr.positions.shape[0], :fr.positions.shape[1]] = fr.positions
                # outA[i, curfrag, :fr.positions.shape[0], :fr.positions.shape[1]] = fr.positions
                # voro = voro3D(ffr[0:8, 0:3], fr.center_of_mass())
            else:
                ffr[:, 0:3] = np.array([fr.positions])
                # outA[i, curfrag, :, 0:3] = fr.positions
                # voro = voro3D(ffr[:, 0:3], fr.center_of_mass())

            ohenc = np.zeros([3])
            ohenc = map(lambda x: x, onehot(typeDict[fr.residues.resnames[0]]))
            ffr[:, 3:6] = ohenc
            # outA[i, curfrag, :, 3:6] = map(lambda x: x, onehot(typeDict[fr.residues.resnames[0]]))

            lipid_beads = fr.names

            for curatom in range(len(fr)):
                bmatrix = np.zeros([len(fr)])
                bead = fr.atoms[curatom]
                ohenc2 = np.zeros([2])
                ohenc2 = map(lambda x: x, onehot(locDict[bead.name], 2))
                ffr[curatom, 6:8] = ohenc2
                # outA[i, curfrag, curatom, 6:8] = map(lambda x: x, onehot(locDict[bead.name], 2))
                curbond = bead.bonds
                # count = 0
                for ib in range(len(curbond)):
                    curblist = curbond.bondlist[ib]
                    curbead = curblist.atoms[0].name
                    # print "beads in bond: ", curblist.atoms[0].name, curblist.atoms[1].name
                    if curbead != bead.name:
                        bmatrix[lipid_beads == curbead] = curblist.length()
                    else:
                        bmatrix[lipid_beads == curblist.atoms[1].name] = curblist.length()

                # ffr[curatom, 8:] = bmatrix
                ffr[curatom, 8:8+len(fr)] = bmatrix
                # print lipid_beads
                # print "Curr bead: ", bmatrix
                # outA[i, curfrag, curatom] = ffr[curatom, :]
                outA[i, curfrag, curatom, :] = ffr[curatom, :]
                # outA[i, curfrag, curatom, 8:8+len(fr)] = bmatrix
                '''
                print "frame_ind: ", frame_ind
                print "curfrag: ", curfrag
                print "curatom: ", curatom
                print "feature vector: ", ffr[curatom, :]
                print "feature vector added: ", outA[frame_ind, curfrag, curatom]
                '''

            # Extrac t and save meighbors ordered by distance
            if fr.resnum in tp.resnums:
                ind = np.argwhere(tp.resnums == fr.resnum)
                _, klist = top_nbrs.get_nbrs_k(ind[0, 0], 50, False)
                current_resnums = tp[klist].resnums
                outNbrs[i, curfrag, :51] = [resnum_dict.get(x, None) for x in current_resnums]
            elif fr.resnum in bt.resnums:
                ind = np.argwhere(bt.resnums == fr.resnum)
                _, klist = bot_nbrs.get_nbrs_k(ind[0, 0], 50, False)
                current_resnums = bt[klist].resnums
                outNbrs[i, curfrag, :51] = [resnum_dict.get(x, None) for x in current_resnums]

            # outL.append([ffr,voro.vertices])

        # outLn = np.array(outL)
        # outAL.append([outLn])
        # outA = np.array(outAL)
        # In case of debug, uncomment the line below
        # print "outA.shape = %s" % str(outA.shape)

        # Flush the frames to disk
        if (i == fchunksize - 1):
            flush_chunk_to_file(processname, i, outA, outNbrs, outResnums, mdsys.opts.outfile, chunkcount, totalchunks)
            i = -1
            chunkcount = chunkcount + 1

        i = i + 1

    if (i != 0):
        # Saves to disk the eventually remaining frames after
        # the last full chunk of data has been written
        flush_chunk_to_file(processname, i, outA, outNbrs, outResnums, mdsys.opts.outfile, chunkcount, totalchunks)


def flush_chunk_to_file(processname, i, outA, outNbrs, outResnums, outfile, chunkcount, totalchunks):
    myfilename = outfile + str(chunkcount).zfill(2) + \
                 "_outof_" + str(totalchunks) + ".npz"
    print "[%d] Flushing chunk (%d records) %d out of %d to file %s" % (processname, i + 1, chunkcount, totalchunks, myfilename)
    # np.save(myfilename, convert_to_helgi_format(outA))
    # np.save(myfilename, outA)
    np.savez_compressed(myfilename, features=outA, neighbors=outNbrs, resnums=outResnums)


def main():
    mdsys = read_mddata()

    # Get leaflets for the simulation
    # top_head, bot_head = find_leaflets(mdsys)

    (totframes, fchunksize, totalchunks) = partition_work(mdsys)

    """Create one process per cpu core and process multiple chunks in parallel
    """
    n = min(mp.cpu_count(), totalchunks)
    queues = []
    processes = []
    chunks_per_task = int(ceil(float(totalchunks) / float(n)))
    print "Using %d ranks and break up the work into %d chunks per task" % (n, chunks_per_task)
    starting_frame = 0
    for i in range(0, n):
        queues.append(Queue())
        ending_frame = min((i+1) * (chunks_per_task * fchunksize), totframes)
        print ending_frame
        starting_chunk = i * chunks_per_task

        mdsys_thread = read_mddata()

        # Get leaflets for the simulation
        top_head, bot_head = find_leaflets(mdsys_thread)

        process = Process(target=process_gromacs_xtc,
                          args=(queues[i], i, totframes, fchunksize, totalchunks,
                                starting_frame, ending_frame, starting_chunk,
                                mdsys_thread, top_head, bot_head))
        processes.append(process)
        starting_frame = ending_frame
        print "Starting process: ", i

    for i in range(0, n):
        processes[i].start()

    for i in range(0, n):
        processes[i].join()


if __name__ == '__main__':
    main()
