# PrepareDataByJson
this project is use to prepare the Faster RCNN Dataset by labelme's json file.  
(You need to prepare your own data)

## 主要實現功能  
### Labelme之資料處理
> 將labelme標註後所輸出的json file進行處理，輸出成Faster RCNN之Dataset形式  
> (範例) :  
<img src="https://github.com/minwei1997/PrepareDataByJson/blob/main/gt_roidb_sample.png" width="550" height="180">  

> 並且將其輸出5個可視化資料  
> (範例):  

![image](https://github.com/minwei1997/PrepareDataByJson/blob/main/json_output_sample.PNG)  

### Data Augmentation
> 實現了三種資料擴增方法(同時處理影像及bounding box):
>> 1. Horizontal flip
>> 2. Vertical flip
>> 3. Rotate

> 資料擴增示意圖 :  

 
-------------------------
## 使用方法
> 將labelme標註之json檔放進 ".\data\js_data\js_file" 路徑  
> 接著執行json_extract.py即可進行處理  
> Faster RCNN之Dataset將存至 ".\data\training_pickle"  
> 5個可視化資料將存至 ".\data\js_data\js_output"  

