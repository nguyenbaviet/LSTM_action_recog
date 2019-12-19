<b>LSTM and Bi-LSTM for action recognition</b>

<br><b>Model type: 1 for LSTM, 2 for Bi-LSTM</b>
<br><b>Dataset: 1 for MERL, 2 for UCI</b>
<br><b>To run</b>
<br>
CUDA_VISIBLE_DEVICES=${GPUs} python exp/main.py --epochs ${num epochs} --batch_size ${batch size} --lr ${learing rate} --dataset ${type of dataset} --hidden_size ${hidden size} --model_vers ${type of model} --folder ${path to save file logs}
