#!/usr/bin/python

import argparse, os
import subprocess

def main(args):
    job = 'rm'
    out = '/dev/null'
    err = '/dev/null'
    flags = '-J {} -o {} -e {} -c {:d} -t {} -p {} {}'.format(job, out, err, args.c, args.t, ','.join(args.p), args.d)
    cmd = 'sbatch {} --wrap="rm {} {}"'.format(flags, '-r' if args.r else '', ' '.join(args.f))
    os.system(cmd)

class ValidatePartition(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        valid_p = ['menon','owners','normal','dev','long','gpu','bigmem']
        for p in values:
            if p not in valid_p:
                raise ValueError('invalid partition {}\nValid partition are: {}'.format(p, ','.join(valid_p)))
        setattr(args, self.dest, values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'submits rm to the SLURM queue on the Sherlock Cluster', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('f', metavar = 'S', type = str, nargs = '+', help = 'files to remove')
    parser.add_argument('-p', metavar = 'P', type = lambda x: str(x).split(","), default = ['menon'], action = ValidatePartition,  help = 'sets the slurm partition, default=menon, options=menon,owners,normal,dev,long,gpu,bigmem')
    parser.add_argument('-t', '--time', dest = 't', metavar = 'T', type = str, nargs = "?", default = '02:00:00', help = 'sets the job time alloted, default=02:00:00')
    parser.add_argument('-m', '--mem', dest = 'm', metavar = 'M', type = str, nargs = "?", default = '32G', help = 'sets the memory for the job, default=32G')
    parser.add_argument('-c', metavar = 'C', type = int, nargs = "?", default = 8, help = 'sets the cpus for the job, default=8')
    parser.add_argument('-r', action = 'store_true', default = 8, help = 'recursive')
    parser.add_argument('-d', metavar = 'D', type = lambda x: '' if x == '' else '--dependency='+str(x), nargs = "?", default = '', help = '''sets the dependency, default=none,
after:jobid[:jobid...]  job can begin after the specified jobs have started,
afterany:jobid[:jobid...] job can begin after the specified jobs have terminated,
afternotok:jobid[:jobid...] job can begin after the specified jobs have failed,
afterok:jobid[:jobid...]  job can begin after the specified jobs have run to completion,
singleton jobs can begin execution after previously launched jobs with the same name and user have ended.''')

	
    args = parser.parse_args()
    main(args)
