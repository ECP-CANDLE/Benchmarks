#!/usr/bin/python
# MDreader
# Copyright (c) Manuel Nuno Melo (m.n.melo@rug.nl)
#
# Released under the GNU Public Licence, v2 or any higher version
#
"""
Class for the all-too-frequent task of asking for options and reading in trajectories. 

version v2016.30.06
by Manuel Melo (m.n.melo@rug.nl) with contribution from Jonathan Barnoud (j.barnoud@rug.n)

The global variable raise_exceptions (default False) controls whether mdreader should handle
end-user errors on its own by exiting cleanly and replacing a traceback by a neater error
message, or instead simply raise the exception and let the user script catch it.
"""

# TODO: account for cases where frames have no time (a GRO trajectory, for example).

from __future__ import division
import sys
import argparse
import os
import numpy as np
import re
import MDAnalysis
import math
import datetime
import types
import multiprocessing
import textwrap


# Globals ##############################################################
########################################################################
# Default is to handle own errors, with a neat exit. Change to allow
#  exceptions to reach the calling code.
raise_exceptions = False
_default_opts = {'s'    : 'topol.tpr',
                 'f'    : 'traj.xtc',
                 'o'    : 'data.xvg',
                 'b'    : 'from the beginning of the trajectory',
                 'e'    : 'until the end of the trajectory',
                 'skip' : 1,
                 'np'   : 0,
                 'v'    : 1}

# Helper functions and decorators ######################################
########################################################################

def _with_defaults(defargs, clobber=True):
    """Decorator to set functions' default arguments

    The decorator takes a dictionary as an argument, which will
    supply default values for all the function arguments that match
    one of its keys.
    
    The decorator doesn't reorder function arguments. This means that, 
    as with def statements, after one argument has been assigned a default
    value all the following arguments must also be given default values.

    The decorator's clobber keyword defines whether existing default values
    are kept or clobbered by values in the passed dictionary.
    """
    def _fnmod(f):
        f_vars = f.__code__.co_varnames[:f.__code__.co_argcount]
        if f.__defaults__:
            ndefs = len(f.__defaults__)
            f_defaults = dict(zip(f_vars[-ndefs:], f.__defaults__)) 
        else:
            f_defaults = {}
        if clobber:
            f_defaults.update(defargs) 
        else:
            f_defaults, f_d = defargs.copy(), f_defaults
            f_defaults.update(f_d)
        new_defaults = []
        for var in f_vars:
            try:
                new_defaults.append(f_defaults[var])
            except KeyError:
                if new_defaults:
                    prev_arg = f_vars[f_vars.index(var)-1]
                    raise TypeError("While attempting to set defaults for the arguments of function "
                            "'{fname}' argument '{arg}' comes after optional argument '{prev_arg}' but was assigned "
                            "no default value. Either set a default value for '{arg}' or modify the base function "
                            "so that '{arg}' comes before any optional arguments.".format(fname=f.func_name, arg=var, prev_arg=prev_arg))
        f.__defaults__ = tuple(new_defaults)
        return f
    return _fnmod

def _do_be_flags(val, default, asframenum):
    if val == default:
        return None
    else:
        if asframenum:
            val = int(val)
        else:
            val = float(val)
        #check_positive(val)
        return val

def _parallel_launcher(rdr, w_id):
    """ Helper function for the parallel execution of registered functions.

    """
    rdr.p_id = w_id
    return rdr._reader()

def _parallel_extractor(rdr, w_id):
    """ Helper function for the parallel extraction of trajectory coordinates/values.

    """
    # block seems to be faster.
    rdr.p_mode = 'block'
    rdr.p_id = w_id
    return rdr._extractor()

def concat_tseries(lst, ret=None):
    """ Concatenates a list of Timeseries objects """
    if ret is None:
        ret = lst[0]
    if len(lst[0]._tjcdx_ndx):
        ret._cdx = np.concatenate([i._cdx for i in lst])
    for attr in ret._props:
        setattr(ret, attr, np.concatenate([getattr(i, attr) for i in lst]))
    return ret

def raise_error(exc, msg):
    if raise_exceptions:
        raise exc(msg)
    else:
        sys.exit("{}: {}".format(exc.__name__, msg))

def check_file(fname):
    if not os.path.exists(fname):
        raise_error(IOError, 'Can\'t find file %s' % (fname))
    if not os.access(fname, os.R_OK):
        raise_error(IOError, 'Permission denied to read file %s' % (fname))
    return fname

def check_outfile(fname):
    dirname = os.path.dirname(fname)
    if not dirname:
        dirname = '.'
    if not os.access(dirname, os.W_OK):
        raise_error(IOError, 'Permission denied to write file %s' % (fname))
    return fname

def check_positive(val, strict=False):
    if strict and val <= 0:
        raise_error(ValueError, "Argument '%r' must be > 0" % (val))
    elif val < 0:
        raise_error(ValueError, "Argument '%r' must be >= 0" % (val))

# Workaround for the lack of datetime.timedelta.total_seconds() in python<2.7
if hasattr(datetime.timedelta, "total_seconds"):
    dtime_seconds = datetime.timedelta.total_seconds
else:
    def dtime_seconds(dtime):
        return dtime.days*86400 + dtime.seconds + dtime.microseconds*1e-6

# Helper Classes #######################################################
########################################################################

class Pool():
    """ MDAnalysis and multiprocessing's map don't play along because of pickling. This solution seems to work fine.

    """
    def __init__(self, processes):
        self.nprocs = processes

    def map(self, f, argtuple):
        procs = []
        nargs = len(argtuple)
        result = [None]*nargs
        arglist = list(argtuple)
        self.outqueue = multiprocessing.Queue()
        freeprocs = self.nprocs
        num = 0
        got = 0
        while arglist:
            while arglist and freeprocs:
                procs.append(multiprocessing.Process(target=self.fcaller, args=((f, arglist.pop(0), num) )))
                num += 1
                freeprocs -= 1
                # procs[-1].daemon = True
                procs[-1].start()
            i, r = self.outqueue.get() # Execution halts here waiting for output after filling the procs.
            result[i] = r
            got += 1
            freeprocs += 1
        # Must wait for remaining procs, otherwise we'll miss their output.
        while got < nargs:
            i, r = self.outqueue.get()
            result[i] = r
            got += 1
        for proc in procs:
            proc.terminate()
        return result

    def fcaller(self, f, args, num):
        self.outqueue.put((num, f(*args)))


class ProperFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """A hackish class to get proper help format from argparse.

    _format_action had to be cloned entirely to allow for indent customization
    because the implementation is not very modular. Probably will break at some
    point when argparse internals change...
    """
    def __init__(self, *args, **kwargs):
        super(ProperFormatter, self).__init__(*args, **kwargs)

    def _split_lines(self, text, width):
        text = text.strip()
        return textwrap.wrap(text, width) 

    def _format_action(self, action):
        # determine the required width and the entry label
        help_position = min(self._action_max_length + 2,
                            self._max_help_position)
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        action_header = self._format_action_invocation(action)


        # ho nelp; start on same line and add a final newline
        if not action.help:
            tup = self._current_indent, '', action_header
            action_header = '%*s%s\n' % tup

        # short action name; start on the same line and pad two spaces
        elif len(action_header) <= action_width:
            tup = self._current_indent, '', action_width, action_header
            action_header = '%*s%-*s  ' % tup
            indent_first = 0

        # long action name; start on the next line
        else:
            tup = self._current_indent, '', action_header
            action_header = '%*s%s\n' % tup
            indent_first = help_position

        # collect the pieces of the action help
        parts = [action_header]

        # if there was help for the action, add lines of help text
        if action.help:
            help_text = self._expand_help(action)
            if "\t" not in help_text:
                indent_first += 8
            help_lines = self._split_lines(help_text, help_width)
            parts.append('%*s%s\n' % (indent_first, '', help_lines[0]))
            for line in help_lines[1:]:
                parts.append('%*s%s\n' % (help_position + 8, '', line))

        # or add a newline if the description doesn't end with one
        elif not action_header.endswith('\n'):
            parts.append('\n')

        # if there are any sub-actions, add their help as well
        for subaction in self._iter_indented_subactions(action):
            parts.append(self._format_action(subaction))

        # return a single string
        return self._join_parts(parts)


class ThenNow:
    def __init__(self, oldval=None, newval=None):
        self.set(oldval, newval)
    def set(self, oldval, newval):
        self.old = oldval
        self.new = newval
    def fill(self, val):
        # Fill variant for the initial case where we have to assign both at initialization.
        self.set(val, val)
    def update(self, val, fill=False):
        if fill:
            self.fill(val)
        else:
            self.old = self.new
            self.new = val


class memoryCheck():
    """Checks memory of a given system
    Lifted from http://doeidoei.wordpress.com/2009/03/22/python-tip-3-checking-available-ram-with-python/

    """
    def __init__(self):
        if sys.platform == "linux2":
            self.value = self.linuxRam()
        elif sys.platform == "darwin":
            self.value = self.macRam()
        elif sys.platform == "win32":
            self.value = self.windowsRam()
        else:
            self.value = float('inf')
            raise EnvironmentError("Memory detection only works with Mac, Win, or Linux. Memory val set to 'inf'.")
 
    def windowsRam(self):
        """Uses Windows API to check RAM in this OS"""
        kernel32 = ctypes.windll.kernel32
        c_ulong = ctypes.c_ulong
        class MEMORYSTATUS(ctypes.Structure):
            _fields_ = [("dwLength", c_ulong),
                        ("dwMemoryLoad", c_ulong),
                        ("dwTotalPhys", c_ulong),
                        ("dwAvailPhys", c_ulong),
                        ("dwTotalPageFile", c_ulong),
                        ("dwAvailPageFile", c_ulong),
                        ("dwTotalVirtual", c_ulong),
                        ("dwAvailVirtual", c_ulong)]
        memoryStatus = MEMORYSTATUS()
        memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
        kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
        return int(memoryStatus.dwTotalPhys/1024**2)
 
    def linuxRam(self):
        """Returns the RAM of a linux system"""
        import subprocess
        process = subprocess.Popen("free -m".split(), stdout=subprocess.PIPE)
        process.poll()
        totalMemory = process.communicate()[0].split("\n")[1].split()[1]
        return int(totalMemory)

    def macRam(self):
        """Returns the RAM of a mac system"""
        import subprocess
        process = subprocess.Popen("vm_stat", stdout=subprocess.PIPE)
        process.poll()
        outpt = process.communicate()[0]
        try:
            totalPages = int(re.search("Pages free:\s*(\d+).", outpt).groups()[0])
            bytperPage = int(re.search("page size of (\d+) bytes", outpt).groups()[0])
        except:
            raise EnvironmentError("Can't detect how much free memory is available from 'vm_stat'.")
        return int(totalPages*bytperPage/(1024**2))


class SeriesCdx():
    """ Placeholder class for a variable behavior of Timeseries.coords"""
    def __init__(self):
        pass


class Timeseries():
    def __getstate__(self):
        statedict = self.__dict__.copy()
        for attr in ["coords","_coords","_cdx_unpacker"]:
            if attr in statedict:
                del statedict[attr]
        return statedict

    def __init__(self):
        self._coords = SeriesCdx()
        self._props = []
        self._tjcdx_ndx = []
        self._tjcdx_relndx = []
        self._cdx = None
        self._xyz = (True, True, True)
        self._coords_istuple = False
        def _cdx_unpacker(n):
            return self._cdx[:,self._tjcdx_relndx[n]]
        self._coords.__getitem__ = _cdx_unpacker

    @property
    def coords(self):
        if self._coords_istuple:
            return self._coords
        else:
            return self._cdx


class DummyParser():
    def __init__(self, *args, **kwargs):
        self._opts = argparse.Namespace()

    def add_argument(self, *args, **kwargs):
        dest = kwargs.get('dest')
        if dest is not None:
            nargs = kwargs.get('nargs')
            if nargs is not None:
                val = [kwargs.get('default')]
            else:
                val = kwargs.get('default')
            setattr(self._opts, dest, val)

    def parse_args(self, *args, **kwargs):
        return self._opts


class _NamedAtlist(np.ndarray):
    """Adds a name to a list of indices, as a property."""
    def __new__(cls, indices, name, attr='ndx_name'):
        if isinstance(indices, cls):
            ret = indices
        if isinstance(indices, np.ndarray):
            ret = indices.view(cls)
        arr = np.array(indices)
        ret = np.ndarray.__new__(cls, shape=arr.shape, dtype=arr.dtype, buffer=arr)
        ret._xtra_attr = attr
        setattr(ret, attr, name)
        return ret
    def to_atgroup(self, univ, name="", attr='ndx_prompt'):
        atgp = univ.atoms[self]
        setattr(atgp, self._xtra_attr, getattr(self, self._xtra_attr))
        if name:
           setattr(atgp, attr, name) 
        return atgp


# MDreader Class #######################################################
########################################################################

# Effectivelly, MDreader will also inherit from either argparse.ArgumentParser
# or from DummyParser.
class MDreader(MDAnalysis.Universe):
    """An object class inheriting from both argparse.ArgumentParser and MDAnalysis.Universe. Should be initialized as for argparse.ArgumentParser, with additional named arguments:
    Argument 'arguments' should be passed the list of command line arguments; it defaults to sys.argv[1:], which is very likely what you'll want.
    Argument 'outstats' defines how often (framewise) to output frame statistics. Defaults to 1.
    Argument 'statavg' defines over how many frames to average statistics. Defaults to 100.
    Argument 'internal_argparse' lets the user choose whether they want to let MDreader take care of option handling. Defaults to True. If set to False, a set of default filenames and most other options (starttime, endtime, etc.) will be used. Check functions setargs and add_ndx on how to change these defaults, or directly modify the mdreader.opts object attributes.

    The command-line argument list will default to:
    
    usage: %prog% [-h] [-f [TRAJ [TRAJ ...]]] [-s TOPOL]
                  [-o OUT] [-b TIME/FRAME]
                  [-e TIME/FRAME] [-fmn] [-skip FRAMES]
                  [-np NPROCS] [-v LEVEL] [-n [INDEX]]

    optional arguments:
      -h, --help                    show this help message and exit
      -f [TRAJ [TRAJ ...]]  file    The trajectory to analyze. If multiple files
                                    they'll be analyzed concatenated. (default: traj.xtc)
      -s TOPOL              file    .tpr, .gro, or .pdb file with the same atom
                                    numbering as the trajectory. (default: topol.tpr)
      -o OUT                file    The main data output file. (default: data.xvg)
      -b TIME/FRAME         real    Time to begin analysis from. If -fmn is set,
                                    -b takes instead an int, as the starting frame number.
                                    (default: from the beginning of the trajectory)
      -e TIME/FRAME         real    Time to end analysis at. If -fmn is set, -e
                                    takes instead an int, as the end frame number.
                                    (default: until the end of the trajectory)
      -fmn                  bool    Whether to interpret -b and -e as frame
                                    numbers (0-based). (default: False)
      -skip FRAMES          int     Interval between frames when analyzing.
                                    (default: 1)
      -np NPROCS            int     Number of processes to parallelize over when
                                    iterating. 1 means serial iteration, and 0 uses the
                                    OS-reported number of cores. Ignored when using MPI,
                                    or when the script specifically sets the number of
                                    parallelization workers. (default: 0)
      -v LEVEL              enum    Verbosity level. 0:quiet, 1:progress 2:debug
                                    (default: 1)
      -n [INDEX]            file    Index file. Defaults to 'index.ndx' if the
                                    filename is not specified. If this flag is omitted
                                    altogether index information will be built from
                                    residue names. (default: None)

    where %prog% is the 'prog' argument as supplied during initialization, or sys.argv[0] if none is provided.
    
    After MDreader instantiation the values of the defaults to the arguments can be changed using MDreader.setargs() (also for setting/unsetting automatic file IO checking; see function documentation). If a 'ver' argument is passed to setargs it will be displayed as the program version, and a '-V'/'--version' option for that purpose will be automatically created.
    The arguments for an MDreader instance can also be added or overridden using the add_argument() method (see the argparse documentation).
    The iterate() method will iterate over the trajectory according to the supplied options, yielding frames as it goes. You'll probably want to use it as part of a for-loop header.
    argparse deprecates using the 'version' argument to __init__. If you need to set it, use the setargs method.
    
    """

    internal_argparse = True

    def __new__(cls, *args, **kwargs):
        bases = (cls,) + cls.__bases__
        try:
            cls.internal_argparse = kwargs['internal_argparse']
        except KeyError:
            pass
        if cls.internal_argparse:
            newcls = type(cls.__name__, bases + (argparse.ArgumentParser,), {})
        else:
            newcls = type(cls.__name__, bases + (DummyParser,), {})
        return super(MDreader, newcls).__new__(newcls)

    def __init__(self, arguments=sys.argv[1:], outstats=1, statavg=100, *args, **kwargs):
        """ Sets up the MDreader object, but doesn't initialize most heavy stuff.

        Option parsing and topology/trajectory loading is left to be done on a need basis.
        keyword 'arguments' allows one to specify a custom list of CLI-like arguments.
        keyword 'outstats' controls how often to report performance statistics.
        keyword 'statavg' controls over how many frames to accumulate performance statistics.
        Finally, keyword 'internal_argparse' allows one to specify whether to use argparse for
        option parsing (set to True) or to use a DummyParser instead (set to False). In the
        latter case one must later supply all needed options by hand, via the setargs method.
        """
        self.arguments = arguments
        # Some users don't like to have argparse thrown in
        #self.internal_argparse is set at the class and __new__ level
        if self.internal_argparse:
            # Set these, unless the user has requested them specifically.
            if len(args) < 10:
                kwargs.setdefault("conflict_handler", 'resolve') 
            if len(args) < 6:
                kwargs.setdefault("formatter_class", ProperFormatter) 
            argparse.ArgumentParser.__init__(self, *args, **kwargs)
            self.check_files = True # Whether to check for readability and writability of input and output files.
        else:
            DummyParser.__init__(self, *args, **kwargs)
            self.check_files = False
        self.version = None
        self.setargs()
        self._parsed = False
        self.hasindex = False
        self._nframes = None
        # Stuff pertaining to progress output/parallelization
        self.parallel = False  # Whether to parallelize
        self.p_smp = False  # SMP parallelization (within the same machine, or virtual machine)
        self.p_mpi = False  # MPI parallelization
        self.outstats = outstats
        self.statavg = statavg
        self.loop_dtimes = np.empty(self.statavg, dtype=datetime.timedelta)
        self.loop_time = ThenNow()
        self.progress = None
        self.framestr = "{1:3.0%}  "
        self.p_mode = 'block'
        self.p_overlap = 0
        self.p_num = None
        self.p_id = 0
        self.p_scale_dt = True
        self.p_mpi_keep_workers_alive = False
        self.p_parms_set = False
        self.i_parms_set = False
        self._cdx_meta = False # Whether to also return time/box arrays when extracting coordinates.

        # Check whether we're running under MPI. Not failsafe, but the user should know better than to fudge with these env vars.
        mpivarlst = ['PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'OMPI_MCA_ns_nds_vpid',
                     'PMI_ID', 'SLURM_PROCID', 'LAMRANK', 'MPI_RANKID',
                     'MP_CHILD', 'MP_RANK', 'MPIRUN_RANK']
        self.mpi = bool(sum([var in os.environ.keys() for var in mpivarlst]))

    # The overridable function for parallel processing.
    def p_fn(self):
        pass

    def __len__(self):
        return self.totalframes

    def __getattr__(self, name):
    # This is a fancy way of only doing important stuff when needed. Also, the user no longer needs to call do_parse, even to access MDAnalysis sub objects.
        if name in ['startframe','endframe','totalframes']:
            try:
                return getattr(self, "_"+name)
            except AttributeError:
                self._set_frameparms()
                return getattr(self, "_"+name)
        if not self._parsed:
            self.do_parse()
            return getattr(self, name)
        else:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, name))

    @property
    def nframes(self):
        if self._nframes is None:
            self.ensure_parsed()
            # Trajectory indexing can be slow. No need to do for every MPI worker: we just pass the offsets around.
            if not self.p_id or not self.mpi:
                self._nframes = len(self.trajectory)
                if self._nframes is None or self._nframes < 1:
                    raise_error(IOError, 'No frames to be read.')
            if self.mpi:
                if hasattr(self.trajectory, "_TrjReader__offsets"):
                    self.trajectory._TrjReader__offsets = self.comm.bcast(self.trajectory._TrjReader__offsets, root=0)
                    if self.p_id != 0:
                        self.trajectory._TrjReader__numframes = len(self.trajectory._TrjReader__offsets)
                        self._nframes = len(self.trajectory._TrjReader__offsets)
                elif self.p_id:
                    self._nframes = len(self.trajectory)
        return self._nframes

    @_with_defaults(_default_opts)
    def setargs(self, s, f, o, b, e, skip, np, v, version=None, check_files=None):
        """ This function allows the modification of the default parameters of the default arguments without having
            to go through the hassle of overriding the args in question. The arguments to this function will override the defaults
            of the corresponding options. These defaults are taken even when internal_argparse has been set to False.
            In the particular case of the 'o' and 'np' arguments one can pass 'None' to hide the option. Check the
            set_parallel_parms method on how to set a specific parallelization number.
            check_files (also accessible via MDreader_obj.check_files) controls whether checks are performed on the readability and
            writabilty of the input/output files defined here (default behavior is to check).
        """
        # Slightly hackish way to avoid code duplication
        parser = self #if self.internal_argparse else self._dummyopts
        # Note: MUST always use dest as a kwarg, to satisfy the DummyParser. Anything without 'dest' will be ignored by it (only relevant when the user sets internal_argparse to False)
        parser.add_argument('-f', metavar='TRAJ', dest="infile", default=f, nargs="*",
                help = 'file\tThe trajectory to analyze. If multiple files they\'ll be analyzed concatenated.')
        parser.add_argument('-s', metavar='TOPOL', dest="topol", default=s,
                help = 'file\t.tpr, .gro, or .pdb file with the same atom numbering as the trajectory.')
        if o is None:
            parser.add_argument('-o', metavar='OUT', dest='outfile', default='data.xvg',
                    help = argparse.SUPPRESS)
        else:
            parser.add_argument('-o', metavar='OUT', dest='outfile', default=o,
                    help = 'file\tThe main data output file.')
        parser.add_argument('-b', metavar='TIME/FRAME', dest='starttime', default=b,
                help = 'real\tTime to begin analysis from. If -fmn is set, -b takes '
                    'instead an int, as the starting frame number.')
        parser.add_argument('-e', metavar='TIME/FRAME', dest='endtime', default=e,
                help = 'real\tTime to end analysis at. If -fmn is set, -e takes '
                    'instead an int, as the end frame number.')
        parser.add_argument('-fmn',  action='store_true', dest='asframenum',
                help = 'bool\tWhether to interpret -b and -e as frame numbers (0-based).')
        parser.add_argument('-skip', metavar='FRAMES', type=int, dest='skip', default=skip,
                help = 'int \tInterval between frames when analyzing.')
        if np is None:
            parser.add_argument('-np', metavar='NPROCS', type=int, dest='parallel', default=_default_opts['np'],
                    help = argparse.SUPPRESS)
        else:
            parser.add_argument('-np', metavar='NPROCS', type=int, dest='parallel', default=np,
                    help = 'int \tNumber of processes to parallelize over when iterating. 1 means serial '
                    'iteration, and 0 uses the OS-reported number of cores. Ignored when using MPI, or when '
                    'the script specifically sets the number of parallelization workers.')
        parser.add_argument('-v', metavar='LEVEL', type=int, choices=[0,1,2], dest='verbose', default=v,
                help = 'enum\tVerbosity level. 0:quiet, 1:progress 2:debug')
        if version is not None:
            parser.add_argument('-V', '--version', action='version', version='%%(prog)s %s'%version,
                help = 'Prints the script version and exits.')
        if check_files is not None:
            self.check_files = check_files

    def add_ndx(self, ng=1, ndxparms=[], ndxdefault='index.ndx', ngdefault=1, smartindex=True):
        """Adds an index read to the MDreader. A -n option will be added.
        ng controls how many groups to ask for.
        If ng is set to 'n' a -ng option will be added, which will then control how many groups to ask for.
        ndxparms should be a list of strings to be printed for each group selection. The default is "Select a group" (a colon is inserted automatically).
        To allow for one or more reference groups plus n analysis groups, ndxparms will be interpreted differently according to ng and the -ng option:
            If ng is "n" it will be set to the number of groups specified by option -ng plus the number of ndxparms elements before the last.
            If ng is greater than the number of elements in ndxparms, then the last element will be repeated to fulfill ng. If ndxparms is greater, all its elements will be used and ng ignored.
        ngdefault sets the default for the -ng option (itself defaulting to 1).
        ndxdefault sets the default for the -n option. Contrary to other mdreader flags, -n can be passed without arguments, in which case it will default to ndxdefault (which is 'index.ndx' by default). If -n is not passed at all it'll take a value of None, and mdreader will then default to getting group info from the system residue names.
        Note: if internal_argparse has been set to False, then ndxdefault directly sets which file to take as the index.
        smartindex controls smart behavior, by which an index with a number of groups equal to n is taken as is without prompting. You'll want to disble it when it makes sense to pick the same index group multiple times, or when order is important.

        Example:
        #Simple index search for a single group. Default message:
        MDreader_obj.add_ndx()
        #Ask for more groups:
        MDreader_obj.add_ndx(ng=3)
        #Ask for a reference group plus a number of analysis groups to be decided with -ng
        MDreader_obj.add_ndx(ng="n", ndxparms=["Select a reference group", "Select a group for doing stuff"])

        """
        if self.hasindex:
            raise AttributeError("Index can only be set once.")
        self.hasindex = True
        parser = self if self.internal_argparse else self.opts
        parser.add_argument('-n', metavar='INDEX', nargs='?', dest='ndx', default=None, const=ndxdefault,
              help = 'file\tIndex file. Defaults to \'%s\' if the filename is not specified. If this flag is omitted altogether index information will be built from residue names.' % ndxdefault)

        self.ng = ng
        if ng == "n":
            if not self.internal_argparse:
                raise ValueError("When setting internal_argparse to False you cannot pass 'n' for the 'ng' parameter of add_ndx (as there is no way for the MDreader object to then ask for the number of index groups). Instead, find out how large 'n' is and pass that number to add_ndx.")
            self.add_argument('-ng', metavar='NGROUPS', type=int, dest='ng', default=ngdefault,
                    help = 'file\tNumber of groups for analysis.')
        self.ndxparms = ndxparms
        self.smartindex = smartindex
        
    def ensure_parsed(self):
        if not self._parsed:
            self.do_parse()

    def do_parse(self):
        """ Parses the command-line arguments according to argparse and does some basic sanity checking on them. It also prepares some argument-dependent loop variables.
        If it hasn't been called so far, do_parse() will be called by the iterate() method, or when trying to access attributes that require it.
        Usually, you'll only want to call this function manually if you want to make sure at which point the arguments are read/parsed.

        """
        self.opts = self.parse_args(self.arguments)
        if self.internal_argparse:
            # We find the version string in the parser _actions. Somewhat fragile.
            for action in self._actions:
                if isinstance(action, argparse._VersionAction) and self.version is None:
                    self.version = action.version
        if self.mpi:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.p_id = self.comm.Get_rank()

        if self.opts.verbose and self.p_id == 0:
            sys.stderr.write("Loading...\n")
        ## Post option handling. outfile and parallel might be unset.
        if isinstance(self.opts.infile, basestring):
            self.opts.infile = [self.opts.infile,]
        if self.check_files:
            map(check_file, [self.opts.topol] + self.opts.infile)
            check_outfile(self.opts.outfile)
        check_positive(self.opts.skip, strict=True)
        check_positive(self.opts.parallel)

        # -b/-e flag handling:
        self.opts.starttime = _do_be_flags(self.opts.starttime, _default_opts['b'], self.opts.asframenum)
        self.opts.endtime = _do_be_flags(self.opts.endtime, _default_opts['e'], self.opts.asframenum)

        #if self.opts.endtime is not None and self.opts.endtime < self.opts.starttime:
        #    raise_error(ValueError, 'Specified end time/frame lower than start time/frame.')

        if not self.p_parms_set:
            self.set_parallel_parms(self.opts.parallel)
        MDAnalysis.Universe.__init__(self, self.opts.topol, *self.opts.infile)

        self.hastime = True
        if not hasattr(self.trajectory.ts, 'time') or self.trajectory.dt == 0.:
            if not self.opts.asframenum and not self.p_id:
                sys.stderr.write("Trajectory has no time information. Will interpret limits as frame numbers.\n")
            self.hastime = False
            self.opts.asframenum = True

        self._parsed = True
        self._set_frameparms()

        if not self.p_id:  # Either there is no MPI, or we're root
            if self.hasindex:
                self._parse_ndx()

        if self.mpi and self.hasindex:   # Get ready to broadcast the index list
            if not self.p_id:
                tmp_ndx = [grp.indices for grp in self.ndxgs]
            else:
                tmp_ndx = None
            tmp_ndx = self.comm.bcast(tmp_ndx, root=0)
            if self.p_id:
                self.ndxgs = [self.atoms[ndx] for ndx in tmp_ndx]

    def _parse_ndx(self):
        self._get__ndx_atgroups()
        self._ndx_prepare()
        self._ndx_input()
        self._select_ndx_atgroups()

    def iterate(self, p=None):
        """Yields snapshots from the trajectory according to the specified start and end boundaries and skip.
        Calculations on AtomSelections will automagically reflect the new snapshot, without needing to refer to it specifically.
        Argument p sets the number of workers, overriding any already set. Note that MDreader is set to use all cores by default, so if you want serial iteration you must pass p=1.
        Other output and parallelization behavior will depend on a number of MDreader properties that are automatically set, but can be changed before invocation of iterate():
          MDreader.progress (default: None) can be one of 'frame', 'pct', 'both', 'empty', or None. It sets the output to frame numbers, %% progress, both, or nothing. If set to None behavior defaults to 'frame', or 'pct' when iterating in parallel block mode.
          MDreader.p_mode (default: 'block') sets either 'interleaved' or 'block' parallel iteration.
          When MDreader.p_mode=='block' MDreader.p_overlap (default: 0) sets how many frames blocks overlap, to allow multi frame analyses (say, an average) to pick up earlier on each block.
          MDreader.p_num (default: None) controls in how many blocks/segments to divide the iteration (the number of workers; will use all the processing cores if set to None) and MDreader.p_id is the id of the current worker for reading and output purposes (to avoid terminal clobbering only p_id 0 will output). 
            **
            If messing with worker numbers (why would you do that?) beware to always set a different p_id per worker when iterating in parallel, otherwise you'll end up with repeated trajectory chunks.
            **
          MDreader.p_scale_dt (default: True) controls whether the reported time per frame will be scaled by the number of workers, in order to provide an effective, albeit estimated, per-frame time.

        """
        self.ensure_parsed()

        if p is not None:
            self.set_parallel_parms(p)
        if not self.p_parms_set:
            self.set_parallel_parms()
        verb = self.opts.verbose and (not self.parallel or self.p_id==0)
        # We're only outputting after each worker has picked up on the pre-averaging frames
        self.i_overlap = True
        self.iterframe = 0

        if not self.i_parms_set:
            self._set_iterparms()

        if verb:
            self._initialize_output_stats()
        # Let's always flush, in case the user likes to print stuff themselves.
        sys.stdout.flush()
        sys.stderr.flush()

        # The LOOP!
        for self.snapshot in self.trajectory[self.i_startframe:self.i_endframe+1:self.i_skip]:
            if self.i_overlap and self.iterframe >= self.p_overlap:
                self.i_overlap = False # Done overlapping. Let the output begin!
            if verb:
                self._output_stats()
            yield self.snapshot
            self.iterframe += 1
        self.i_parms_set = False
        self.p_parms_set = False

    def _initialize_output_stats(self):
        # Should be run before _output_stats, but not absolutely mandatory.
        sys.stderr.write("Iterating through trajectory...\n")

        if self.progress is None:
            if self.parallel and self.p_mode == "block":
                self.progress = 'pct'
            else:
                self.progress = 'frame'
        # Python-style implementation of a switch/case. It also avoids always comparing the flag every frame.
        if self.progress == "frame":
            self.framestr = "Frame {0:d}"
            if self.hastime:
                self.framestr += "  t= {2:.1f} ps  "
        elif self.progress == "pct":
            self.framestr = "{1:3.0%}  "
        elif self.progress == "both":
            self.framestr = "Frame {0:d}, {1:3.0%}"
            if self.hastime:
                self.framestr += "  t= {2:.1f} ps  "
        elif self.progress == "empty":
            self.framestr = ""
        else:
            raise ValueError("Unrecognized progress mode \"%r\"" % (self.progress))


    def _output_stats(self):
        """Keeps and outputs performance stats.
        """
        self.loop_time.update(datetime.datetime.now())
        if self.iterframe: # No point in calculating delta times on iterframe 0
            self.loop_dtime = self.loop_time.new - self.loop_time.old
            self.loop_dtimes[(self.iterframe-1) % self.statavg] = self.loop_dtime
            # Output stats every outstats step or at the last frame.
            if (not self.iterframe % self.outstats) or self.iterframe == self.i_totalframes-1:
                avgframes = min(self.iterframe,self.statavg)
                self.loop_sumtime = self.loop_dtimes[:avgframes].sum()
                # No float*dt multiplication before python 3. Let's scale the comparing seconds and set the dt ourselves.
                etaseconds = dtime_seconds(self.loop_sumtime)*(self.i_totalframes-self.iterframe)/avgframes
                eta = datetime.timedelta(seconds=etaseconds)
                if etaseconds > 300:
                    etastr = (datetime.datetime.now()+eta).strftime("Will end %Y-%m-%d at %H:%M:%S.")
                else:
                    etastr = "Will end in %ds." % round(etaseconds)
                loop_dtime_s = dtime_seconds(self.loop_dtime)
                if self.parallel:
                    if self.p_scale_dt:
                        loop_dtime_s /= self.p_num

                if self.hastime:
                    progstr = self.framestr.format(self.snapshot.frame-1, (self.iterframe+1)/self.i_totalframes, self.snapshot.time)
                else:
                    progstr = self.framestr.format(self.snapshot.frame-1, (self.iterframe+1)/self.i_totalframes)

                sys.stderr.write("\033[K%s(%.4f s/frame) \t%s\r" % (progstr, loop_dtime_s, etastr))
                if self.iterframe == self.i_totalframes-1: 
                    #Last frame. Clean up.
                    sys.stderr.write("\n")
                sys.stderr.flush()
    
    def timeseries(self, coords=None, props=None, x=True, y=True, z=True, parallel=True):
        """Extracts coordinates and/or other time-dependent attributes from a trajectory.
        'coords' can be an AtomGroup, an int, a selection text, or a tuple of these. In case of an int, it will be taken as the mdreader index group number to use.
        'props' must be a str or a tuple of str, which will be used as attributes to extract from the trajectory's timesteps. These must be valid attributes of the mdreader.trajectory.ts class, and cannot be bound functions or reserved '__...__' attributes.
        Will return a mdreader.Timeseries object, holding an array, or a tuple, for each coords, and having named properties holding the same-named time-arrays. If both coords and props are are None the default is to return the time-coordinates array for the entire set of atoms.
        'props' attributes should be set in quotes, ommitting the object's name.
        'parallel' (default=True) controls parallelization behavior.
        'x', 'y', and 'z' (default=True) set whether the three coordinates, or only a subset, are extracted.

        Examples:
        1. Timeseries with whole time-coordinate array for all atoms:
        mdreader.timeseries()

        2. Equivalent to above:
        mdreader.timeseries(mdreader.atoms)

        3. Timeseries with a tuple of two selections time-coordinates (the NC3 atoms, and the fifth chosen group from the index):
        mdreader.timeseries(("name NC3", 4))

        4. Timeseries with an array of box time-coordinates:
        mdreader.timeseries(props='dimensions')

        5. Timeseries with a time-coordinate array correspondig to the x,y components of the second index group, and time-arrays of the system's box dimensions and time.
        mdreader.timeseries(coords=1, props=('dimensions', 'time'), z=False)

        """
        # First things first
        self.ensure_parsed()

        self._tseries = Timeseries()
        tjcdx_atgrps = []
        if coords is None and props is None:
            tjcdx_atgrps = [self.atoms]
        elif coords is not None:
            if type(coords) == MDAnalysis.core.AtomGroup.AtomGroup:
                tjcdx_atgrps = [coords]
            elif type(coords) == types.IntType:
                tjcdx_atgrps = [self.ndxgs[coords]]
            elif isinstance(coords, basestring):
                tjcdx_atgrps = [self.select_atoms(coords)]
            else:
                self._tseries._coords_istuple = True
                try:
                    for atgrp in coords:
                        if type(atgrp) == types.IntType:
                            tjcdx_atgrps.append(self.ndxgs[atgrp])
                        elif type(atgrp) == MDAnalysis.core.AtomGroup.AtomGroup:
                            tjcdx_atgrps.append(atgrp)
                        else:
                            tjcdx_atgrps.append(self.select_atoms("%s" % atgrp))
                except:
                    raise TypeError("Error parsing coordinate groups.\n%r" % sys.exc_info()[1])

        # Get the unique list of indices, and the pointers to that list for each requested group.
        indices = [grp.indices for grp in tjcdx_atgrps]
        indices_len = [len(ndx) for ndx in indices]
        self._tseries._tjcdx_ndx, self._tseries._tjcdx_relndx = np.unique(np.concatenate(indices), return_inverse=True)
        self._tseries._tjcdx_relndx = np.split(self._tseries._tjcdx_relndx, np.cumsum(indices_len[:-1])) 

        self._tseries._xyz = (x,y,z)
        mem = self.atoms[self._tseries._tjcdx_ndx].coordinates()[0].nbytes*sum(self._tseries._xyz)

        if props is not None:
            if isinstance(props, basestring):
                props = [props]
            self._tseries._props = []
            #validkeys = self.trajectory.ts.__dict__.keys()
            for attr in props:
                if not hasattr(self.trajectory.ts, attr):
                    raise AttributeError('Invalid attribute for extraction. It is not an attribute of trajectory.ts')
                self._tseries._props.append(attr)
                # Rough memory checking
                mem += sys.getsizeof(getattr(self.trajectory.ts, attr))
                setattr(self._tseries, attr, None)
        mem *= len(self)

        # This is potentially a lot of memory. Check it beforehand, except for MPI, which we trust the user to do themselves.
        if not self.p_mpi:
            avail_mem = memoryCheck()
            if 2*mem/(1024**2) > avail_mem.value:
                raise_error(EnvironmentError, "You are attempting to read approximately %d MB of coordinates/values but your system only seems to have %d MB of physical memory (and we need at least twice as much memory as read bytes)." % (mem/(1024**2), avail_mem.value))

        tseries = self._tseries
        if not self.p_smp:
            tseries = self._extractor()
            if self.p_mpi:
                tseries = self.comm.gather(tseries, root=0)
                if self.p_id == 0:
                    tseries = concat_tseries(tseries)
        else:
            pool = Pool(processes=self.p_num)
            concat_tseries(pool.map(_parallel_extractor, [(self, i) for i in range(self.p_num)]), tseries)

        if self.p_mpi and not self.p_mpi_keep_workers_alive and self.p_id != 0:
            sys.exit(0)
        else:
            self._tseries = None
            tseries.atgrps = tjcdx_atgrps
            self.p_parms_set = False
            return tseries


    def do_in_parallel(self, fn, *args, **kwargs):
        """ Applies fn to every frame, taking care of parallelization details. Returns a list with the returned elements, in order.
        args and kwargs should be an iterable, resp. a dictionary, of arguments that will be passed (with the star, resp. double-star, operator) to fn. Default to the empty tuple and empty dict.
        parallel can be set to False to force serial behavior. Setting it to True forces default parallelization behavior, overriding previous settings of self.p_num.
        Refer to the documentation on MDreader.iterate() for information on which MDreader attributes to set to change default parallelization options.

        """
        self.p_fn = fn
        try:
            parallel = kwargs.pop("parallel")
        except KeyError:
            force_p_recheck = False
        else:
            nprocs = int(not parallel) # parallel=True (resp. False) becomes nprocs=0 (resp. 1)
            force_p_recheck = True
        self.p_args = args
        self.p_kwargs = kwargs

        self.ensure_parsed()
        if force_p_recheck:
            self.set_parallel_parms(nprocs)

        if not self.p_smp:
            if not self.p_mpi:
                return self._reader()
            else:
                res = self._reader()
                res = self.comm.gather(res, root=0)
                if not (self.p_id == 0 or self.p_mpi_keep_workers_alive):
                    sys.exit(0)
        else:
            pool = Pool(processes=self.p_num)
            res = pool.map(_parallel_launcher, [(self, i) for i in range(self.p_num)]) 

        # 1-level unravelling and de-interlacing
        if self.p_smp or (self.p_mpi and self.p_id == 0):
            if self.p_mode == "block":
                return [val for subl in res for val in subl] 
            elif self.p_mode == "interleaved":
                ret = []
                for ctr in range(len(res[0])):
                    for subl in res:
                        try:
                            ret.append(subl[ctr])
                        except IndexError:
                            pass
                return ret
            else:
                raise NotImplementedError("Unknown parallelization mode '%s'" % self.p_mode)

    def _reader(self):
        """ Applies self.p_fn for every trajectory frame. Parallelizable!

        """

        if self.p_smp:
        # We need a brand new file descriptor per SMP worker, otherwise we have a nice chaos.
        # This must be the first thing after entering parallel land.
            self._reopen_traj()

        reslist = []
        if not self.i_parms_set:
            self._set_iterparms()
        if self.i_unemployed: # This little piggy stays home
            self.i_parms_set = False
            self.p_parms_set = False
            return reslist

        for frame in self.iterate():
            result = self.p_fn(*self.p_args, **self.p_kwargs)
            if not self.i_overlap:
                reslist.append(result)
        return reslist

    def _extractor(self):
        """ Extracts the values asked for in mdreader._tseries. Parallelizable!

        """
        # This should become a function to pass to _reader... Lots of code duplication between these two.

        if self.p_smp:# and self.p_id: ROOT IS NOT EXEMPT! Multiprocessing starts a new process for root too.
        # We need a brand new file descriptor per SMP worker, otherwise we have a nice chaos.
        # This must be the first thing after entering parallel land.
            self._reopen_traj()

        if not self.i_parms_set:
            self._set_iterparms()

        if len(self._tseries._tjcdx_ndx):
            self._tseries._cdx = np.empty((self.i_totalframes, len(self._tseries._tjcdx_ndx), sum(self._tseries._xyz)), dtype=np.float32)
        for attr in self._tseries._props:
            try:
                shape = (self.i_totalframes,) + getattr(self.trajectory.ts, attr).shape
            except AttributeError:
                shape = (self.i_totalframes,)
            try:
                setattr(self._tseries, attr, np.empty(shape, dtype=(getattr(self.trajectory.ts, attr)).dtype))
            except AttributeError:
                setattr(self._tseries, attr, np.empty(shape, dtype=type(getattr(self.trajectory.ts, attr))))
        if not self.i_unemployed:
            for frame in self.iterate():
                if self._tseries._cdx is not None:
                    self._tseries._cdx[self.iterframe] = self.atoms[self._tseries._tjcdx_ndx].coordinates()[:,np.where(self._tseries._xyz)[0]]
                for attr in self._tseries._props:
                    getattr(self._tseries, attr)[self.iterframe,...] = getattr(self.trajectory.ts, attr)
        return self._tseries
    
    def _reopen_traj(self):
       # Let's make this generic and always loop over a list of formats. If it's the ChainReader then they all get in.
       rdrs = []
       if self.trajectory.format == "CHAIN":
           rdrs.extend(self.trajectory.readers)
       else:
           rdrs.append(self.trajectory)
       for rdr in rdrs:
           # XTC/TRR reader has this method, but not all formats...
           if hasattr(rdr, "_reopen"):
               rdr._reopen()
           elif hasattr(rdr, "dcdfile"):
               rdr.dcdfile.close()
               rdr.dcdfile = open(self.trajectory.filename, 'rb')
           else:
               raise_error(AttributeError, "Don't know how to get a new file descriptor for the %s trajectory format. You'll have to skip parallelization." % rdr.format)

    def _set_frameparms(self):
        if self.opts.asframenum:
            if self.opts.starttime is None:
                self._startframe = 0
            elif self.opts.starttime < 0:
                self._startframe = self.nframes + self.opts.starttime
            else:
                self._startframe = self.opts.starttime
            #
            if self.opts.endtime is None:
                self._endframe = self.nframes-1
            elif self.opts.endtime < 0:
                self._endframe = self.nframes + self.opts.starttime
            else:
                self._endframe = min(self.nframes-1, self.opts.endtime)
        else:
            # We bend over backwards here with np.rint to ensure the correct int ceil (python 2.7 doesn't have it yet).
            t0 = self.trajectory[0].time
            if self.opts.starttime is None:
                self._startframe = 0
            elif self.opts.starttime < 0.:
                self._startframe = self.nframes + int(math.ceil(self.opts.starttime/self.trajectory.dt)) 
            elif t0 - self.opts.starttime > 1e-7:
                raise_error(ValueError, "You requested to start at time %f but the trajectory "
                                        "starts already at time %f." % (self.opts.starttime, t0))
            else:
                self._startframe = int(np.rint(math.ceil((self.opts.starttime-t0)/self.trajectory.dt)))
            #
            if self.opts.endtime is None:
                self._endframe = self.nframes-1
            elif self.opts.endtime < 0.:
                self._endframe = self.nframes + int(math.ceil(self.opts.endtime/self.trajectory.dt)) 
            elif t0 - self.opts.endtime > 1e-7:
                raise_error(ValueError, "Specified end time lower (%f ps) than the trajectory start time (%f ps)." % (self.opts.endtime, t0))
            else:
                self._endframe = min(int(np.rint(math.ceil((self.opts.endtime-t0)/self.trajectory.dt))), self.nframes-1)

        if self._startframe >= self.nframes:
            if self.opts.asframenum:
                raise_error(ValueError, "You requested to start at frame %d but the trajectory only has %d frames." % (self.opts.starttime, self.nframes))
            else:
                raise_error(ValueError, "You requested to start at time %f ps but the trajectory only goes up to %f ps." % (self.opts.starttime, (self.nframes-1)*self.trajectory.dt))
        if self._endframe < self._startframe:
            raise_error(ValueError, 'Specified end time/frame lower than start time/frame.')
        if self._endframe < 0 or self._startframe < 0:
            raise_error(ValueError, 'Resulting start/end frame lower than 0.')

        self._totalframes = int(np.rint(math.ceil(float(self._endframe - self._startframe+1)/self.opts.skip)))

    def _set_iterparms(self):
        # Because of parallelization lots of stuff become limited to the iteration scope.
        # defined a group of i_ variables just for that.
        self.i_unemployed = False
        if self.parallel:
            #if self.p_num < 2 and self.p_smp:
            #    raise ValueError("Parallel iteration requested, but only one worker (MDreader.p_num) sent to work.")

            #self.i_startframe and self.i_endframe must be specifically cast as
            # ints because of an unpythonic type check in MDAnalysis that misses numpy ints.
            if self.p_mode == "interleaved":
                frames_per_worker = np.ones(self.p_num,dtype=np.int)*(self.totalframes//self.p_num)
                frames_per_worker[:self.totalframes%self.p_num] += 1 # Last workers to arrive work less. That's life for you.
                self.i_skip = self.opts.skip * self.p_num
                self.i_startframe = int(self.startframe + self.opts.skip*self.p_id)
                self.i_endframe = int(self.i_startframe + int(frames_per_worker[self.p_id]-1)*self.i_skip)
            elif self.p_mode == "block":
                # As-even-as-possible distribution of frames per workers, allowing the first one to work more to compensate the lack of overlap.
                frames_per_worker = np.ones(self.p_num,dtype=np.int)*((self.totalframes-self.p_overlap)//self.p_num)
                frames_per_worker[:(self.totalframes-self.p_overlap)%self.p_num] += 1 
                frames_per_worker[0] += self.p_overlap # Add extra overlap frames to the first worker.
                self.i_skip = self.opts.skip
                self.i_startframe = int(self.startframe + np.sum(frames_per_worker[:self.p_id])*self.i_skip)
                self.i_endframe = int(self.i_startframe + (frames_per_worker[self.p_id]-1)*self.i_skip)
                # And now we subtract the overlap from the startframe, except for worker 0
                if self.p_id:
                    self.i_startframe -= self.p_overlap*self.i_skip
            else:
                raise ValueError("Unrecognized p_mode \"%r\"" % (self.p_mode))
            # Let's check for zero work
            if not frames_per_worker[self.p_id]:
                self.i_unemployed = True
        else:
            self.i_skip = self.opts.skip
            self.i_startframe = self.startframe
            self.i_endframe = self.endframe
        self.i_totalframes = int(np.rint(math.ceil((self.i_endframe-self.i_startframe+1)/self.i_skip)))
        self.i_parms_set = True


    def set_parallel_parms(self, nprocs=None):
        """Resets parallelization parameters. Use to specifiy the number of processes.
        The nprocs argument sets how many processors to use. 0 defaults to the OS-reported
        number, and 1 sets up serial iteration. If nprocs is left at None, then the last
        used number of processors will be re-used (behaving like nprocs=0 if not yet set).
        """
        if self.p_num is None or nprocs is not None:
            self.p_num = nprocs
        self.parallel = self.p_num != 1
        self.p_mpi = self.parallel and self.mpi
        self.p_smp = self.parallel and not self.mpi
        if self.parallel:
            if self.p_mpi:
                self.p_num = self.comm.Get_size() # MPI size always overrides manually set p_num. The user controls the pool size with mpirun -np nprocs
            elif self.p_smp and not self.p_num:
                self.p_num = multiprocessing.cpu_count()
        if self.p_num == 1: # For single-core machines, or single-process MPI runs
            self.parallel = False
            self.p_mpi = False
            self.p_smp = False
        self.p_parms_set = True

    def info_header(self, line_prefix=''):
        self.ensure_parsed()
        header = "{}{} {}\n".format(line_prefix, self.prog, " ".join(self.arguments))
        if self.version is not None:
            header = "{}{} {}\n".format(line_prefix, self.prog, self.version) + header
        return header

    def _get__ndx_atgroups(self):
        if self.opts.ndx is not None:
            self._ndx_atlists=[]
            tmpstr=""
            ndxheader=None
            with open(self.opts.ndx) as NDX:
                while True:
                    line = NDX.readline()
                    mtx = re.match('\s*\[\s*(\S+)\s*\]\s*',line)
                    if mtx or not line:
                        if ndxheader is not None:
                            self._ndx_atlists.append(_NamedAtlist(np.array(tmpstr.split(), dtype=int)-1, ndxheader))
                            tmpstr = ""
                        if not line:
                            break
                        ndxheader = mtx.groups()[0]
                    else:
                        tmpstr += line
        else:
            resnames = np.unique(self.atoms.resnames)
            self._ndx_atlists = [_NamedAtlist(self.atoms.indices, "System")]
            self._ndx_atlists.extend([_NamedAtlist(self.select_atoms("resname %s" % (rn,)).indices, rn) for rn in resnames ])
        self._ndx_names = [ndx.ndx_name for ndx in self._ndx_atlists]

    def _ndx_prepare(self):
        """Prepares number and content of index prompt strings. Decides on whether to autoassign index groups."""
        # How many groups to auto assign (it may be useful to have the same group as a reference and as an analysis group, so we check it a bit more thoroughly).
        if self.ng == "n":
            self._refng = max(0,len(self.ndxparms)-1)
            otherng = self.opts.ng
            self.ng = self._refng+otherng
        elif self.ng > len(self.ndxparms):
            self._refng = max(0,len(self.ndxparms)-1)
            otherng = self.ng-self._refng
        else:
            self.ng = len(self.ndxparms)
            otherng = self.ng
            self._refng = 0

        if not self.ndxparms:
            self.ndxparms = ["Select a group"]*self.ng
        elif self.ng > len(self.ndxparms):
            self.ndxparms.extend([self.ndxparms[-1]]*(otherng-1))

        if self.smartindex and len(self._ndx_atlists)==otherng:
            self._autondx = otherng
        else:
            self._autondx = 0

    def _ndx_input(self):
        """Prepares self.ndx_stdin to read from piped text or interactive prompt."""
        # Check for interactivity, otherwise just eat it from stdin
        self.ndx_stdin = []
        if sys.stdin.isatty():
            self.interactive = True
            maxlen = str(max(map(lambda x:len(x.ndx_name), self._ndx_atlists)))
            maxidlen = str(len(str(len(self._ndx_atlists)-1)))
            maxlenlen = str(max(map(len, (map(str, (map(len, self._ndx_atlists)))))))
            for ndxgid, hd in enumerate(self._ndx_atlists):
                sys.stderr.write(("Group %"+maxidlen+"d (%"+maxlen+"s) has %"+maxlenlen+"d elements\n") % (ndxgid, hd.ndx_name, len(hd)))
            sys.stderr.write("\n")
        else:
            self.interactive = False
            import select
            if not self.mpi or select.select([sys.stdin], [], [], 0)[0]: # MPI is tricky because it blocks stdin.readlines()
                self.ndx_stdin = "".join(sys.stdin.readlines()).split()

    def _select_ndx_atgroups(self):
        self.ndxgs=[]
        auto_id = 0       # for auto assignment of group ids
        for gid, ndxprompt in enumerate(self.ndxparms):
            if gid < self._refng or not self._autondx:
                if not self.ndx_stdin:
                    sys.stderr.write("%s:\n" % (ndxprompt))
                ndx_str = self._getinputline()
                try: # First try as an integer
                    self.ndxgs.append(self._ndx_atlists[int(ndx_str)].to_atgroup(self, ndxprompt))
                except ValueError:
                    try: # Now as the group's name
                        self.ndxgs.append(self._ndx_atlists[self._ndx_names.index(ndx_str)].to_atgroup(self, ndxprompt))
                    except ValueError:
                        raise_error(KeyError, "Group name %s not found in index." % (ndx_str))
            else:
                if gid == self._refng:
                    sys.stderr.write("Only %d groups in index file. Reading them all.\n" % len(self._ndx_atlists))
                self.ndxgs.append(self._ndx_atlists[auto_id].to_atgroup(self, ndxprompt))
                auto_id += 1

    def _getinputline(self):
        while True:
            if self.ndx_stdin:
                return self.ndx_stdin.pop(0)
            elif self.interactive:
                self.ndx_stdin.extend(raw_input().split())
            else:
                raise_error(IndexError, "\nNo (or not enough) index groups were passed to stdin. If you're running under MPI make sure to pipe in the group numbers; for instance:\n$ echo 2 4 6 | mpirun script.py\nor\n$ mpirun script.py < file_with_list_of_groups")


class SimpleReader(MDreader):
    """A shortcut class inheriting from MDreader. Creates an MDreader instance without argparse inheriting.
    The arguments set the values for MDreader.opts (which won't be asked from the user) and possibly also an index.
    The following arguments (followed by their defaults), correspond to the flags asked by the MDreader parser:
        s='topol.tpr', f='traj.xtc', o='data.xvg', b=0, e=float('inf'), skip=1, v=1
    The following arguments (followed by their defaults) will be passed to the add_ndx function. add_ndx will only be called if ng or ndxparms is set:
        ndx=None, ndxparms=None, ng=None, smartindex=True
    Argument check_files (default=None) has the same meaning as for the setargs function.
    """

    internal_argparse=False

    def __init__(self, s='topol.tpr', f='traj.xtc', o='data.xvg', b=0, e=float('inf'), skip=1, v=1, check_files=None, ndx=None, ndxparms=None, ng=None, smartindex=True):
        super(SimpleReader, self).__init__() 
        self.setargs(s=s, f=f, o=o, b=b, e=e, skip=skip, v=v, version=None, check_files=check_files)
        if ndxparms or ng:
            self.add_ndx(ndxparms=ndxparms, ndxdefault=ndx, ng=ng, smartindex=smartindex)


class DefaultReader(MDreader):
    """A shortcut class inheriting from MDreader. Creates an MDreader instance with default values.
    The arguments allow the and possibility of automatic addition of an index.
    If any arguments are present, they'll be passed to add_ndx, for the creation of an index. Refer to that function's documentation for the relevant arguments.
    """
    def __init__(self, *args, **kwargs):
        super(DefaultReader, self).__init__() 
        if args or kwargs:
            self.add_ndx(*args, **kwargs)

