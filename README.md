# yolov7_on_snpe

兩個資料夾是以不同的model轉成的DLC，一個是Yolov7(input_size為640x640)，一個是Yolov7-tiny。
其中Yolov7-tiny中再分成三種，model的input_size分別為640x640，416x416，以及自己訓練的模型(input_size為416x416，預測的class剩下23個)
首先要先建立SNPE的環境，可以參考SNPE環境建置.md
建立完成之後，就是套用DLC進行AI辨識物件的部分，可以參考Code使用.md
