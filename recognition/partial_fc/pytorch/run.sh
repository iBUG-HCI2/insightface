# /usr/bin/zsh
export OMP_NUM_THREADS=4
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s)
end_seconds=$(date --date="$endtime" +%s)
starttime=`date +'%Y-%m-%d %H:%M:%S'`
python3 -m torch.distributed.launch --nproc_per_node=8 partial_fc.py --world_size=8 | tee hist.log
echo "Running time:"$((end_seconds-start_seconds))"s"
ps -ef | grep "world_size" | grep -v grep | awk '{print "kill -9 "$2}' | sh