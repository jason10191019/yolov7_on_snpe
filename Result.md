以下是我所這次所嘗試使用的model
從最一開始用yolov7 standard開始，
到最後使用yolov7_tiny並縮減model圖片尺寸與預測的classes，
使得運算時間逐漸降低。
1.yolov7 standard 640*640 運行時間
![image](https://github.com/jason10191019/yolov7_on_snpe/assets/80830129/878909d6-9e30-4985-932a-7846f690f131)

2.yolov7_tiny 640*640 運行時間
![image](https://github.com/jason10191019/yolov7_on_snpe/assets/80830129/ee3242fb-0bca-44ab-8205-4c44c515e2da)

3.yolov7_tiny 416*416運行時間
![螢幕擷取畫面 2023-12-05 173908](https://github.com/jason10191019/yolov7_on_snpe/assets/80830129/489a6da1-f8f0-4cbf-baa1-b803d0c52d85)

4.yolov7_tiny 416*416(23classes)運行時間
![螢幕擷取畫面 2023-12-05 173459](https://github.com/jason10191019/yolov7_on_snpe/assets/80830129/dab0575a-a6d6-4ab5-89f6-f2eea8969b5c)

5.yolov7_tiny 416*416(9classes)運行時間
![螢幕擷取畫面 2023-12-05 173115](https://github.com/jason10191019/yolov7_on_snpe/assets/80830129/129020ab-a005-4b65-875d-7de45ed5417d)














