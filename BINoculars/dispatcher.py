import os
import time
import itertools
import subprocess
import multiprocessing

from . import util, errors, space


class Destination(object):
    type = filename = overwrite = value = config = None
    opts = {}
    
    def set_final_filename(self, filename, overwrite):
        self.type = 'final'
        self.filename = filename
        self.overwrite = overwrite

    def set_final_options(self, opts):
        if opts is not False:
            self.opts = opts

    def set_config(self, conf):
        self.config = conf

    def set_tmp_filename(self, filename):
        self.type = 'tmp'
        self.filename = filename

    def set_memory(self):
        self.type = 'memory'

    def store(self, space):
        self.value = None
        if self.type == 'memory':
            self.value = space
        elif self.type == 'tmp':
            space.tofile(self.filename)
        elif self.type == 'final':
            fn = self.final_filename()
            space.config = self.config
            space.tofile(fn)

    def retrieve(self):
        if self.type == 'memory':
            return self.value

    def final_filename(self):
        fn = self.filename.format(**self.opts)
        if not self.overwrite:
            fn = util.find_unused_filename(fn)
        return fn

class DispatcherBase(util.ConfigurableObject):
    def __init__(self, config, main):
        self.main = main
        super(DispatcherBase, self).__init__(config)

    def parse_config(self, config):
        super(DispatcherBase, self).parse_config(config)
        self.config.destination = Destination()
        self.config.destination.set_final_filename(
            config.pop('destination', 'output.hdf5'),# optional 'output.hdf5' by default
            util.parse_bool(config.pop('overwrite', 'false')))#by default: numbered files in the form output_###.hdf5:

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
        return space.chunked_sum(results)


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
        self.config.ncores = int(config.pop('ncores', 0))# optionally, specify number of cores (autodetect by default)
        if self.config.ncores <= 0:
            self.config.ncores = multiprocessing.cpu_count()

    def process_jobs(self, jobs):
        if self.config.ncores == 1: # note: SingleCore will be marginally faster
            imap = itertools.imap
        else:
            pool = multiprocessing.Pool(self.config.ncores)
            imap = pool.imap_unordered

        configs = (self.prepare_config(job) for job in jobs)
        for result in imap(self.main.get_reentrant(), configs):
            yield result

    def sum(self, results):
        return space.chunked_sum(results)

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
        self.config.tmpdir = config.pop('tmpdir', os.getcwd())#Optional, current directory by default
        self.config.oarsub_options = config.pop('oarsub_options', 'walltime=0:15')# optionally, tweak oarsub parameters
        self.config.executable = config.pop('executable', ' '.join(util.get_python_executable()))# optionally, override default location of python and/or BINoculars installation

    def process_jobs(self, jobs):
        self.configfiles = []
        self.intermediates = []
        clusters = list(util.cluster_jobs(jobs, self.main.input.config.target_weight))
        for i, jobscluster in enumerate(clusters, start=1):
            uniq = util.uniqid()
            jobconfig = os.path.join(self.config.tmpdir, 'binoculars-{0}-jobcfg.zpi'.format(uniq))
            self.configfiles.append(jobconfig)

            config = self.main.clone_config()
            if i == len(clusters):
                config.dispatcher.sum = self.intermediates
            else:
                interm = os.path.join(self.config.tmpdir, 'binoculars-{0}-jobout.hdf5'.format(uniq))
                self.intermediates.append(interm)
                config.dispatcher.destination.set_tmp_filename(interm)
                config.dispatcher.sum = ()

            config.dispatcher.action = 'process'
            config.dispatcher.jobs = jobscluster
            util.zpi_save(config, jobconfig)
            
            yield self.oarsub(jobconfig)

    def sum(self, results):
        jobs = list(results)
        self.oarwait(jobs)

        # cleanup:
        for f in itertools.chain(self.configfiles, self.intermediates):
            try:
                os.remove(f)
            except Exception as e:
                print "unable to remove {0}: {1}".format(f, e)
        return True

    def run_specific_task(self, command):
        if self.config.action != 'process' or (not self.config.jobs and not self.config.sum) or command:
            raise errors.SubprocessError("invalid command, too many parameters or no jobs/sum given")

        jobs = sum = space.EmptySpace()
        if self.config.jobs:
            jobs = space.sum(self.main.process_job(job) for job in self.config.jobs)
        if self.config.sum:
            sum = space.chunked_sum(space.Space.fromfile(src) for src in self.yield_when_exists(self.config.sum))
        self.config.destination.store(jobs + sum)

    ### UTILITY
    @staticmethod
    def yield_when_exists(files):
        files = set(files)
        polltime = 0
        while files:
            if time.time() - polltime < 5:
                time.sleep(time.time() - polltime)
            polltime = time.time()
            exists = set(f for f in files if os.path.exists(f))
            for e in exists:
                yield e
            files -= exists

    @staticmethod
    def subprocess_run(*command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, unused_err = process.communicate()
        retcode = process.poll()
        return retcode, output

    ### calling OAR
    def oarsub(self, *args):
        command = '{0} process {1}'.format(self.config.executable, ' '.join(args))
        ret, output = self.subprocess_run('oarsub', '-l {0}'.format(self.config.oarsub_options), command)
        if ret == 0:
            lines = output.split('\n')
            for line in lines:
                if line.startswith('OAR_JOB_ID='):
                    void, jobid = line.split('=')
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
                else: # assume state == 'Finishing' or 'Terminated' but don't wait on something unknown
                    del jobs[i]
                    i -= 1 # otherwise it skips a job
                i += 1
            util.status('{0}: {1} jobs to go. {2} waiting, {3} running, {4} unknown.'.format(time.ctime(),len(jobs),W,R,U))
        util.statuseol()
