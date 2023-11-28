# Code 解析

snpe在使用上可以分成三個部分，第一步是建立snpe model，第二步是建立input_tensor，第三步是execute。
這邊我使用的model是Yolov7，input_size是416*416如果模型有調整要記得對其他部分進行修改

# 建立SNPE model
    
```cpp=
zdl::DlSystem::Runtime_t runt=checkRuntime();
std::unique_ptr<zdl::SNPE::SNPE> snpe = initializeSNPE(runt);
```
    
首先先介紹checkRuntime()的部分，這個部分主要是設定使用GPU或是CPU運算。

```cpp=
zdl::DlSystem::Runtime_t checkRuntime()
{
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    static zdl::DlSystem::Runtime_t Runtime;
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
        Runtime = zdl::DlSystem::Runtime_t::GPU;
        std::cout<<"using GPU"<<std::endl;
    } else {
        Runtime = zdl::DlSystem::Runtime_t::CPU;
        std::cout<<"using CPU"<<std::endl;
    }
    return Runtime;
}
```

接著是initializeSNPE的部分，這裡有比較多需要注意的地方，首先是outputLayers，因為我們有自己去切output的nodes，所以這裡要去設置，不然就會預設只有最後一層的output。
首先是outputLayers就是上一篇中Conv的name，三個都設置完之後就可以得到所有layers運算完後的結果。
設置完成之後再讀取DLC模型就完成SNPE model的設置了。

```cpp=
std::unique_ptr<zdl::SNPE::SNPE> initializeSNPE(zdl::DlSystem::Runtime_t runtime) {
    zdl::DlSystem::StringList outputLayers = {};
    outputLayers.append("Conv_296");
    outputLayers.append("Conv_310");
    outputLayers.append("Conv_324");
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open("/root/models/yolov7.dlc");   
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    std::unique_ptr<zdl::SNPE::SNPE> snpe = snpeBuilder.setOutputLayers(outputLayers)
                      .setRuntimeProcessor(runtime)
                      .setCPUFallbackMode(true)
                      .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                      .build();                  
  return snpe;
}
```

# 建立input_tensor

SNPE提供了兩種讀取input的方法，一種是使用buffer，另外一種就是我這邊使用的tensor。

```cpp=
std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe, pixel);
static zdl::DlSystem::TensorMap inputTensorMap;
const auto strList = snpe->getInputTensorNames();
inputTensorMap.add((*strList).at(0), inputTensor.get());
```

首先是loadInputTensor的部分，這裡的code可以在SNPE官網中的SDK找到，要記得先將input處理成vector<float>的方式，這樣就可以完成input tensor的轉換。轉換完成後在將input tensor轉成tensormap就算完成資料處理了。
    
```cpp=
std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor(std::unique_ptr<zdl::SNPE::SNPE> &snpe, std::vector<float> inputVec) {
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    const auto &strList_opt = snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &strList = *strList_opt;

    const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
    const auto &inputShape = *inputDims_opt;
    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);

    std::copy(inputVec.begin(), inputVec.end(), input->begin());
    return input;
}
```
    
# Execute

```cpp=
std::vector<float> output = executeNetwork(snpe, inputTensorMap);
```
    
這段過程也是可以在SNPE官網中的SDK找到，其中我會將output的name印出來是跟post process的部分有關。從印出來的部分我可以知道順序是會先得到32x32的grid預測出的結果，再來是8x8，最後是16x16
    
```cpp=
std::vector<float> executeNetwork(std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                zdl::DlSystem::TensorMap inputTensorMap) {
    static zdl::DlSystem::TensorMap outputTensorMap;
    snpe->execute(inputTensorMap, outputTensorMap);
    std::vector<float> output;
    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
    std::for_each( tensorNames.begin(), tensorNames.end(), [&](const char* name) {
        std::cout<<name<<std::endl;
        auto tensorPtr = outputTensorMap.getTensor(name);
        for ( auto it = tensorPtr->cbegin(); it != tensorPtr->cend(); ++it ){
            float f = *it;
            output.push_back(f);
        }
    });

    return output;
}
```
    
# Post Process
    
上面已經完成了SNPE模型的建置以及運算，但因為原本的yolov7會在reshape成5-d layers後再進行sigmoid等運算，不過我們將output切在reshape之前，所以需要自己實作這部分。
    
### 為什麼原本會需要reshape
    
以416*416的圖片為例，yolov7的模型中有三種大小的grid分別是32x32，16x16，8x8，每個grid會預測出三個bounding box，每一個grid的output大小為5+num_classes，在yolov7中num_clases共有80個。output中前四個分別是x座標，y座標，寬和高，第五個是confidence score，後面則是各個class的class score。所以32x32的grid就會得到1x13x13x225大小的output，而16x16就會得到1x26x15x225，8x8則會得到1x52x52x225。Reshape層的作用就是先將225的部分分成3x85後，在進行sigmoid等處理。
![unnamed](https://hackmd.io/_uploads/H1dValXH6.png)
    
### 整理output
    
因為每個bounding box的大小都是5+num_classes，如果是yolov7就是5+80，所以我先將output以每個bounding_box為一組進行整理。
    
```cpp=
int groupsize = 85;
int group = output.size()/groupsize;
std::vector<std>::vector<float>> box(group, std::vector<float(groupsize));

for(size_t i=0;i<output.size();i++){
    int x = i/groupsize;
    int y = i%groupsize;
    box[x][y] = output[i];
}
```
        
### Sigmoid
    
因為我們把output提前切出來，所以得到的數據還需要先進行sigmoid。
如果是x座標，y座標，寬和高還要再根據anchorGrid算出實際座標位置與長度
anchorGrid可以在yolov7中的cfg/training找到，根據使用的模型進行調整，如果是tiny的話就換成另外一組數據，處理過程如下
    
```cpp=
std::vector<std::vector<float>> sigmoid(std::vector<std::vector<float>> in){
    float strides[3] = {32, 8, 16};
    float anchorGrid[][6] = {         //到cfg/training中查看
        {142,110, 192,243, 459,401},  // 32*32
        {12,16, 19,36, 40,28},        // 8*8
        {36,75, 76,55, 72,146},      // 16*16
    };
    float size[3]={13,52,26};
    //如果model的img_size有更動的話，這裡就改成{size/32,size/8,size/16}
    // copy all outputs to one array.
    // [13 * 13 * 3 * 85]----\
    // [52 * 52 * 3 * 85]--------> [10647 * 85]
    // [26 * 26 * 3 * 85]----/

    for (size_t i = 0; i < 3; i++) {
        int height = size[i];
        int width = size[i];
        for (int j = 0; j < height; j++) {      // 13/52/26
            for (int k = 0; k < width; k++) {   // 13/52/26
                int anchorIdx = 0;
                for (int l = 0; l < 3; l++) {   // 3
                    int index = 3*(size[i]*j+k)+l;
                    if(i==1){
                        index+=507;  //507(13*13*3)
                    }else if(i==2){
                        index+=8619;  //8619(13*13*3+52*52*3)
                    }

                    for (int m = 0; m < 85; m++) {     // 85
                        if (m < 2) {                   // x,y座標算法
                            in[index][m]=1.0/(1.0+exp(static_cast<double>(-in[index][m])));
                            float value = in[index][m];
                            float gridValue = m == 0 ? k : j;
                            in[index][m] = (value * 2 - 0.5 + gridValue) * strides[i];
                        } else if (m < 4) {            //寬,高算法
                            in[index][m]=1.0/(1.0+exp(static_cast<double>(-in[index][m])));
                            float value = in[index][m];
                            in[index][m] = value * value * 4 * anchorGrid[i][anchorIdx++];
                        } else{                        //confidence_score與class_score算法
                            in[index][m]=1.0/(1.0+exp(static_cast<double>(-in[index][m])));
                        }
                    }
                }
            }
        }
    }
    return in;
} 
```
    
### NMS
    
等進行完以上處理後可以先對confidence score進行篩選，這邊我設的值是0.3，如果有大於的話，再找出class score最高的項目後，與confidence score相乘，如果還是大於0.3的話就是通過篩選的目標bounding box。
    
不過這樣可能會導致一個物體有很多bounding box，因此我們要使用NMS來過濾掉多餘的bounding box。

首先我會先按照confidence score大小進行排序，接著就可以從confidence score最高的開始，檢測其他bounding box是否跟目前的bounding_box偵測了同一個物體，如果是同一個物體就將其他bounding box刪掉。
檢測方法則是透過比較bounding box的重合面積。
    
```cpp=
bool compareByFifthElement(const std::vector<float>& a, const std::vector<float>& b) {
    return a[4] > b[4];
}

std::vector<std::vector<float>> nms(std::vector<std::vector<float>> input,float thres){
    std::sort(input.begin(), input.end(),compareByFifthElement);
    std::vector<bool> flag(input.size(), false);
    for(size_t i=0;i<input.size();i++) {
        if (flag[i]) continue;
        for (unsigned int j = i + 1; j < input.size(); j++) {
            if (calculateIoU(input[i], input[j]) > thres) flag[j] = true;
        }
    }
    std::vector<std::vector<float>> output;
    for (size_t i = 0; i < input.size(); i++) {
        if (!flag[i]) 
            output.push_back(input[i]);
    }
    return output;
}
```

比較重合面積的實作過程，這邊要注意的是從模型中得到的x,y座標是中心點，所以在計算時記得要先做計算。
     
```cpp=
float calculateIoU(std::vector<float> input1, std::vector<float> input2) {
    // 计算边界框的坐标
    if(input1[5]!=input2[5]){
        return 0;
    }
    float x1 = std::max(input1[0] - input1[2]/2 , input2[0] - input2[2]/2);
    float y1 = std::max(input1[1] - input1[3]/2 , input2[1] - input2[3]/2);
    float x2 = std::min(input1[0] + input1[2]/2 , input2[0] + input2[2]/2);
    float y2 = std::min(input1[1] + input1[3]/2 , input2[1] + input2[3]/2);

    // 计算交集区域的面积
    float intersectionArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

    // 计算并集区域的面积
    float unionArea = input1[2] * input1[3] + input2[2] * input2[3] - intersectionArea;

    // 计算 IoU
    return intersectionArea / unionArea;
}
```
    
# 畫出bounding box

經過剛剛的篩選之後，現在就剩下最後一步了，就是只要把bounding box畫出來！
    畫圖時一樣要注意從模型中得到的x,y座標是中心點的座標，而cv2畫圖時會需要輸入左上角的x,y座標，因此要記得先算出左上角的座標。方法如下：
    
```cpp=
cv::Scalar color(0, 255, 0);
for(size_t i=0;i<result.size();i++){
    cv::Rect bbox;
    bbox.width=result[i][2];
    bbox.height=result[i][3];
    bbox.x=result[i][0]-bbox.width/2;
    bbox.y=result[i][1]-bbox.height/2;
    cv::rectangle(image_new, bbox, color, 2);  // 2 是边框的宽度
    int cf = result[i][4]*100;
    std::string className = ClassMap[result[i][5]]+std::to_string(cf)+"%";
    cv::putText(image_new, className, cv::Point(bbox.x, bbox.y -5),
                cv::FONT_HERSHEY_SIMPLEX, 1 , cv::Scalar(0, 255, 0), 2 , cv::LINE_AA );
}
    cv::putText(image_new, std::to_string(result.size()),cv::Point (10,30),
                cv::FONT_HERSHEY_SIMPLEX, 1 , cv::Scalar(0, 255, 0), 2 , cv::LINE_AA );
```