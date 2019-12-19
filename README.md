#LSTM and Bi-LSTM for action recognition

#Model type: 1 for LSTM, 2 for Bi-LSTM
#Dataset: 1 for MERL, 2 for UCI
#To run
CUDA_VISIBLE_DEVICES=${GPUs} python exp/main.py --epochs ${num epochs} --batch_size ${batch size} --lr ${learing rate} --dataset ${type of dataset} --hidden_size ${hidden size} --model_vers ${type of model} --folder ${path to save file logs}
