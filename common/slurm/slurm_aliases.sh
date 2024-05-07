export slurmscript=$OAK/projects/astrock/2021_common/scripts/slurm
alias sourcenv='[ -f ./environment.sh ] && { source ./environment.sh; echo "Starting project $PROJECT_NAME"; } || { echo "Setup your environment before running."; }'
export flag1c='-c 1 -p normal,menon'
export flag8c='-c 8 -p normal,menon'
export flag1g='-c 8 -G 1 -p menon,normal,gpu'
export flag4g='-c 8 -G 4 -N 1 -p menon,normal,gpu'
export flagtime='10:00:00'
export flagmem='50G'

alias srm='python $slurmscript/rmsubmit.py --time=$flagtime $flag1c,owners'
alias smv='python $slurmscript/mvsubmit.py --time=$flagtime $flag1c,owners'

alias submit1c='python $slurmscript/pysubmit.py --time=$flagtime --mem=$flagmem $flag1c,owners'
alias submit8c='python $slurmscript/pysubmit.py --time=$flagtime --mem=$flagmem $flag8c,owners'
alias submit1g='python $slurmscript/pysubmit.py --time=$flagtime --mem=$flagmem $flag1g,owners'
alias submit4g='python $slurmscript/pysubmit.py --time=$flagtime --mem=$flagmem $flag4g,owners --Gshared'
alias printflag='echo --time=$flagtime --mem=$flagmem'