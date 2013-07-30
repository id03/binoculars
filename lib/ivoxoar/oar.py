import subprocess

from . import util


def run(*command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output, unused_err = process.communicate()
    retcode = process.poll()
    return retcode, output


def oarsub(*args, **kwargs):
    opts = kwargs.pop('options', 'walltime=0:15')
    if kwargs:
        raise ValueError('invalid keyword parameter(s): {0}'.format(kwargs))
    scriptname = './blisspython /users/onderwaa/iVoxOar/iVoxOar.py '
    command = '{0} {1}'.format(scriptname, ' '.join(args))
    ret, output = run('oarsub', '-l {0}'.format(opts), command)
    if ret == 0:
        lines = output.split('\n')
        for line in lines:
            if line.startswith('OAR_JOB_ID='):
                void, jobid = line.split('=')
                return jobid
    return False


def oarstat(jobid):
    # % oarstat -s -j 5651374
    # 5651374: Running
    # % oarstat -s -j 5651374
    # 5651374: Finishing
    ret, output = run('oarstat', '-s', '-j', str(jobid))
    if ret == 0:
        for n in output.split('\n'):
            if n.startswith(str(jobid)):
                job, status = n.split(':')
        return status.strip()
    else:
        return 'Unknown'


def oarwait(jobs, remaining=0):
    linelen = 0
    if len(jobs) > remaining:
        util.status('{0}: getting status of {1} jobs...'.format(time.ctime(), len(jobs)))
    else:
        return
 
    while 1:
        i = 0
        R = 0
        W = 0
        U = 0
        while i < len(jobs):
            state = oarstat(jobs[i])
            if state == 'Running':
                R += 1
            elif state == 'Waiting':
                W += 1
            elif state == 'Unknown':
                U += 1
            else: # assume state == 'Finishing' or 'Terminated' but don't wait on something unknown
                del jobs[i]
                i -= 1 #otherwise it skips a job
            i += 1
        util.status('{0}: {1} jobs to go. {2} waiting, {3} running, {4} unknown.'.format(time.ctime(),len(jobs),W,R,U))
        if len(jobs) <= remaining:
            util.statuseol()
            return
        else:
            time.sleep(30) # only sleep if we're not done yet


