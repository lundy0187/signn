# set variables
export SIGNN_MODEL=deepsig
export SIGNN_FOLDER=utils/dataset/gnuradio_sim
export SIGNN_FILE=SIGNN_2019_01_1024.hdf5
export SIGNN_SNR="-10 -6 -2 2 6 10"
export CPU_ONLY="--cpu"
export PREDICT="--predict"
export ONNXCONV="--onnx-conv"

# execute run
#echo $SIGNN_MODEL "Start : "  $(date) >> tuner_log.txt; 
python3 signn_tuner.py -m $SIGNN_MODEL -p $SIGNN_FOLDER -d $SIGNN_FILE --dataset-shape 2 1024 -s artifacts --test --snr $SIGNN_SNR $ONNXCONV $CPU_ONLY $PREDICT; 
#echo $SIGNN_MODEL "End   : "  $(date) >> tuner_log.txt;