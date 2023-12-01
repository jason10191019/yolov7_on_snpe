# SNPE Setup
我使用的環境是在python3.6下，首先先建立這次的虛擬環境，建立完成後依序輸入以下指令

    // setup environment
    # apt-get update
    // create list of alternatives for python
    # update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
    # update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2

    // check the python list
    # update-alternatives --list python
    // update alternative to select required python version
    # update-alternatives --config python

    // upgrade pip
    # apt install python3-pip
    # apt install python-pip
    # pip install --upgrade pip
    # pip -V

    // install the SNPE SDK dependencies
    # apt-get install python3-dev python3-matplotlib python3-numpy python3-protobuf python3-scipy python3-skimage python3-sphinx python3-mako wget zip
    # apt-get install libc++-9-dev

接下來要先下載好兩個檔案snpe-1.68.0.zip與android-ndk-r17c-linux-x86_64.zip並解壓縮

    // verify that all dependencies are installed
    // data/snpe-sdk是資料夾snpe-1.68.0的路徑
    // data/android-ndk-r17c-linux-x86_64是android-ndk-r17c-linux-x86_64資料夾的路徑
    # export ANDROID_NDK_ROOT=data/android-ndk-r17c-linux-x86_64/
    # source data/snpe-sdk/bin/dependencies.sh
    // verify that the Python dependencies are installed
    # source data/snpe-sdk/bin/check_python_depends.sh

    # apt-get install python-protobuf
    # apt-get install cmake
    # apt-get install libprotobuf-dev protobuf-compiler
    //在這步之前先下載好onnx資源包，雖然snpe官方說要使用1.6版本，但建議安裝1.8版本，這樣也可以進行yolov7模型的轉換
    # source data/snpe-sdk/bin/envsetup.sh -o snpe/lib/python3.6/site-packages/onnx/

# Model Transform

以上環境建立完成後就可以開始進行模型的轉換了，我們的目標是把pt檔案轉成dlc檔案，首先可以先下載yolov7的offical weight，如果之後覺得想縮短模型的運算時間可以考慮選擇yolov7-tiny或是減少預測的classes後自行再train一次。

https://github.com/WongKinYiu/yolov7

這是yolov7的github，可以照著裡面的Export部分將pt檔轉成onnx檔案，其中--end2end因為snpe不支援，所以要記得拿掉

    // python export.py --weights yolov7.pt --grid --simplify --img-size 640 640
    # img_size是模型讀取img的尺寸，可以選擇416*416或640*640
    
如果之後想要自己train的話除了按照Training的指令以外，還需要更改其他文件，這部分留到yolov7的部分再介紹。

得到onnx檔案後就可以進行最後DLC檔案的轉換了，因為snpe目前不支援5-D layers的運算，所以我們要先用netron找出要切的node，並在之後自己完成post-process的部分。

![image](https://hackmd.io/_uploads/BkxvmRfrT.png)

從Reshape之後可以看到輸出變成5-d了 因此我們要把output切在Reshape之前，也就是Conv的output，從最右邊的表中可以知道Conv的name是Conv_296，output的name是489，旁邊還有兩個點，一樣也可以得到他們的name。Conv的name會在之後用到，所以這邊可以先記起來，這樣之後就不用再找一次。
接下來就可以做DLC檔案的轉換了，轉換方法如下，這樣就完成從pt模型到DLC模型的轉換了！
    
    //snpe-onnx-to-dlc --input_network yolov7.onnx --output_path yolov7.dlc --out_node 489 --out_node 524 --out_node 559
    

    
