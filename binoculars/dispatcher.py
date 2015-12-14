import sys
import os
import time
import itertools
import subprocess
import multiprocessing

from . import util, errors, space

#python3 support
PY3 = sys.version_info > (3,)

class Destination(object):
    type = filename = overwrite = value = config = limits = None
    opts = {}

    def set_final_filename(self, filename, overwrite):
        self.type = 'final'
        self.filename = filename
        self.overwrite = overwrite

    def set_final_options(self, opts):
        if opts is not False:
            self.opts = opts

    def set_limits(self, limits):
        self.limits = limits

    def set_config(self, conf):
        self.config = conf

    def set_tmp_filename(self, filename):
        self.type = 'tmp'
        self.filename = filename

    def set_memory(self):
        self.type = 'memory'

    def store(self, verse):
        self.value = None
        if verse.dimension == 0:
            raise ValueError('Empty output, Multiverse contains no spaces')
        if self.type == 'memory':
            self.value = verse
        elif self.type == 'tmp':
            verse.tofile(self.filename)
        elif self.type == 'final':
            for sp, fn in zip(verse.spaces, self.final_filenames()):
                sp.config = self.config
                sp.tofile(fn)

    def retrieve(self):
        if self.type == 'memory':
            return self.value

    def final_filenames(self):
        fns = []
        if not self.limits == None:
            base, ext = os.path.splitext(self.filename)
            for limlabel in util.limit_to_filelabel(self.limits):
                fn = (base + '_' + limlabel + ext).format(**self.opts)
                if not self.overwrite:
                    fn = util.find_unused_filename(fn)
                fns.append(fn)
        else:
            fn = self.filename.format(**self.opts)
            if not self.overwrite:
                fn = util.find_unused_filename(fn)
            fns.append(fn)
        return fns


class DispatcherBase(util.ConfigurableObject):
    def __init__(self, config, main):
        self.main = main
        super(DispatcherBase, self).__init__(config)

    def parse_config(self, config):
        super(DispatcherBase, self).parse_config(config)
        self.config.destination = Destination()
        destination = config.pop('destination', 'output.hdf5')  # optional 'output.hdf5' by default
        overwrite = util.parse_bool(config.pop('overwrite', 'false'))  #by default: numbered files in the form output_  # .hdf5:
        self.config.destination.set_final_filename(destination, overwrite)  # explicitly parsing the options first helps with the debugging
        self.config.host = config.pop('host', None)  # ip adress of the running gui awaiting the spaces
        self.config.port = config.pop('port', None)  # port of the running gui awaiting the spaces
        self.config.send_to_gui = util.parse_bool(config.pop('send_to_gui', 'false'))  # previewing the data, if true, also specify host and port

    def send(self, verses):  # provides the possiblity to send the results to the gui over the network
        if self.config.send_to_gui or (self.config.host is not None and self.config.host is not None):  # only continue of ip is specified and send_to_server is flagged
            for M in verses:
                if self.config.destination.limits is None:
                    sp = M.spaces[0]
                    if isinstance(sp, space.Space):
                        util.socket_send(self.config.host, int(self.config.port), util.serialize(sp, ','.join(self.main.config.command)))
                else:
                    for sp, label in zip(M.spaces, util.limit_to_filelabel(self.config.destination.limits)):
                        if isinstance(sp, space.Space):
                            util.socket_send(self.config.host, int(self.config.port), util.serialize(sp, '{0}_{1}'.format(','.join(self.main.config.command), label)))
                yield M
        else:
            for M in verses:
                yield M

    def has_specific_task(self):
        return False

    def process_jobs(self, jobs):
        raise NotImplementedError

    def sum(self, results):
        raise NotImplementedError


# The simplest possible dispatcher. Does the work all by itself on a single
# thread/core/node. 'Local' will most likely suit your needs better.
class SingleCore(DispatcherBase):
    def process_jobs(self, jobs):
        for job in jobs:
            yield self.main.process_job(job)

    def sum(self, results):
        return space.chunked_sum(self.send(results))


# Base class for Dispatchers using subprocesses to do some work.
class ReentrantBase(DispatcherBase):
    actions = 'user',

    def parse_config(self, config):
        super(ReentrantBase, self).parse_config(config)
        self.config.action = config.pop('action', 'user').lower()
        if self.config.action not in self.actions:
            raise errors.ConfigError('action {0} not recognized for {1}'.format(self.config.action, self.__class__.__name__))

    def has_specific_task(self):
        if self.config.action == 'user':
            return False
        else:
            return True

    def run_specific_task(self, command):
        raise NotImplementedError


# Dispatch multiple worker processes locally, while doing the summation in the main process
class Local(ReentrantBase):
    ### OFFICIAL API
    actions = 'user', 'job'

    def parse_config(self, config):
        super(Local, self).parse_config(config)
        self.config.ncores = int(config.pop('ncores', 0))  # optionally, specify number of cores (autodetect by default)
        if self.config.ncores <= 0:
            self.config.ncores = multiprocessing.cpu_count()

    def process_jobs(self, jobs):
        if self.config.ncores == 1 and not PY3:  # note: SingleCore will be marginally faster
            map = itertools.imap
        else:
            pool = multiprocessing.Pool(self.config.ncores)
            map = pool.imap_unordered

        configs = (self.prepare_config(job) for job in jobs)
        for result in map(self.main.get_reentrant(), configs):
            yield result

    def sum(self, results):
        return space.chunked_sum(self.send(results))

    def run_specific_task(self, command):
        if command:
            raise errors.SubprocessError("invalid command, too many parameters: '{0}'".format(command))
        if self.config.action == 'job':
            result = self.main.process_job(self.config.job)
            self.config.destination.store(result)

    ### UTILITY
    def prepare_config(self, job):
        config = self.main.clone_config()
        config.dispatcher.destination.set_memory()
        config.dispatcher.action = 'job'
        config.dispatcher.job = job
        return config, ()

# Dispatch many worker processes on an Oar cluster.


class Oar(ReentrantBase):
    ### OFFICIAL API
    actions = 'user', 'process'

    def parse_config(self, config):
        super(Oar, self).parse_config(config)
        self.config.tmpdir = config.pop('tmpdir', os.getcwd())  # Optional, current directory by default
        self.config.oarsub_options = config.pop('oarsub_options', 'walltime=0:15')  # optionally, tweak oarsub parameters
        self.config.executable = config.pop('executable', ' '.join(util.get_python_executable()))  # optionally, override default location of python and/or BINoculars installation

    def process_jobs(self, jobs):
        self.configfiles = []
        self.intermediates = []
        clusters = util.cluster_jobs2(jobs, self.main.input.config.target_weight)
        for jobscluster in clusters:
            uniq = util.uniqid()
            jobconfig = os.path.join(self.config.tmpdir, 'binoculars-{0}-jobcfg.zpi'.format(uniq))
            self.configfiles.append(jobconfig)

            config = self.main.clone_config()
            interm = os.path.join(self.config.tmpdir, 'binoculars-{0}-jobout.hdf5'.format(uniq))
            self.intermediates.append(interm)
            config.dispatcher.destination.set_tmp_filename(interm)
            config.dispatcher.sum = ()

            config.dispatcher.action = 'process'
            config.dispatcher.jobs = jobscluster
            util.zpi_save(config, jobconfig)
            yield self.oarsub(jobconfig)

        #if all jobs are sent to the cluster send the process that sums all other jobs
        uniq = util.uniqid()
        jobconfig = os.path.join(self.config.tmpdir, 'binoculars-{0}-jobcfg.zpi'.format(uniq))
        self.configfiles.append(jobconfig)
        config = self.main.clone_config()
        config.dispatcher.sum = self.intermediates
        config.dispatcher.action = 'process'
        config.dispatcher.jobs = ()
        util.zpi_save(config, jobconfig)
        yield self.oarsub(jobconfig)

    def sum(self, results):
        jobs = list(results)
        jobscopy = jobs[:]
        self.oarwait(jobs)
        self.oar_cleanup(jobscopy)
        return True

    def run_specific_task(self, command):
        if self.config.action != 'process' or (not self.config.jobs and not self.config.sum) or command:
            raise errors.SubprocessError("invalid command, too many parameters or no jobs/sum given")

        jobs = sum = space.EmptyVerse()
        if self.config.jobs:
            jobs = space.verse_sum(self.send(self.main.process_job(job) for job in self.config.jobs))
        if self.config.sum:
            sum = space.chunked_sum(space.Multiverse.fromfile(src) for src in util.yield_when_exists(self.config.sum))
        self.config.destination.store(jobs + sum)

    ### calling OAR
    @staticmethod
    def subprocess_run(*command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, unused_err = process.communicate()
        retcode = process.poll()
        return retcode, output

    def oarsub(self, *args):
        command = '{0} process {1}'.format(self.config.executable, ' '.join(args))
        ret, output = self.subprocess_run('oarsub', '-l {0}'.format(self.config.oarsub_options), command)
        if ret == 0:
            lines = output.split('\n')
            for line in lines:
                if line.startswith('OAR_JOB_ID='):
                    void, jobid = line.split('=')
                    util.status('{0}: Launched job {1}'.format(time.ctime(), jobid))
                    return jobid.strip()
        return False

    def oarstat(self, jobid):
        # % oarstat -s -j 5651374
        # 5651374: Running
        # % oarstat -s -j 5651374
        # 5651374: Finishing
        ret, output = self.subprocess_run('oarstat', '-s', '-j', str(jobid))
        if ret == 0:
            for n in output.split('\n'):
                if n.startswith(str(jobid)):
                    job, status = n.split(':')
            return status.strip()
        else:
            return 'Unknown'

    def oarwait(self, jobs, remaining=0):
        linelen = 0
        if len(jobs) > remaining:
            util.status('{0}: getting status of {1} jobs...'.format(time.ctime(), len(jobs)))
        else:
            return

        delay = util.loop_delayer(30)
        while len(jobs) > remaining:
            next(delay)
            i = 0
            R = 0
            W = 0
            U = 0

            while i < len(jobs):
                state = self.oarstat(jobs[i])
                if state == 'Running':
                    R += 1
                elif state in ('Waiting', 'toLaunch', 'Launching'):
                    W += 1
                elif state == 'Unknown':
                    U += 1
                else:  # assume state == 'Finishing' or 'Terminated' but don't wait on something unknown
                    del jobs[i]
                    i -= 1  # otherwise it skips a job
                i += 1
            util.status('{0}: {1} jobs to go. {2} waiting, {3} running, {4} unknown.'.format(time.ctime(), len(jobs), W, R, U))
        util.statuseol()

    def oar_cleanup(self, jobs):
        # cleanup:
        for f in itertools.chain(self.configfiles, self.intermediates):
            try:
                os.remove(f)
            except Exception as e:
                print("unable to remove {0}: {1}".format(f, e))

        errorfn = []

        for jobid in jobs:
            errorfilename = 'OAR.{0}.stderr'.format(jobid)

            if os.path.exists(errorfilename):
                with open(errorfilename, 'r') as fp:
                    errormsg = fp.read()
                if len(errormsg) > 0:
                    errorfn.append(errorfilename)
                    print('Critical error: OAR Job {0} failed with the following error: \n{1}'.format(jobid, errormsg))

        if len(errorfn) > 0:
            print('Warning! {0} job(s) failed. See above for the details or the error log files: {1}'.format(len(errorfn), ', '.join(errorfn)))
