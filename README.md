# PrepareDataByJson
this project is use to prepare the Faster RCNN Dataset by labelme's json file.  
(You need to prepare your own data)

## 主要實現功能  
### Labelme之資料處理
> 將labelme標註後所輸出的json file進行處理，輸出成Faster RCNN之xml檔案格式  
> (範例) :  

<img src="https://github.com/minwei1997/PrepareDataByJson/blob/main/%E7%85%A7%E7%89%87/Sample/xml_sample.png" width="336" height="324">  

> 並且將其輸出5個可視化資料  
> (範例):  

![image](https://github.com/minwei1997/PrepareDataByJson/blob/main/%E7%85%A7%E7%89%87/Sample/json_output_sample.PNG)  

### Data Augmentation
> 實現了三種資料擴增方法(同時處理影像及bounding box):
>> 1. Horizontal flip
>> 2. Vertical flip
>> 3. Rotate

> 資料擴增示意圖 :  
<img src="https://github.com/minwei1997/PrepareDataByJson/blob/main/%E7%85%A7%E7%89%87/Data%20Aug/Summary.PNG" width="550" height="114">  
 
-------------------------
## 使用方法
### Json file 檔案處理
> 1. 將瑕疵照片放進 ".\data\Defect_Img" 路徑
> 2. 將labelme標註之json檔放進 ".\data\js_data\js_file" 路徑  
> 3. 執行json_extract.py即可進行處理  
>> Faster RCNN需用到的xml檔將存至 ".\data\xml_file"  
>> 5個可視化資料將存至 ".\data\js_data\js_output"  

### Data Augmentation
> 執行Data_Aug.py即可進行Augmentation，在__main__中需要設定模式(visulize or augment)以及圖片張數
>> 共有兩個版本(1為gt_roidb格式的，2為xml格式的(翻轉+Rotate)，3為xml格式的(先做完翻轉再全部做Rotate))
>>> 1. Data_Aug
>>> 2. Data_Aug_v2 
>>> 2. Data_Aug_v3 

-------------------------
## 附屬功能
> 1. PngToJpg -> 將圖片的副檔名變更(可由程式內自行修改要變更的副檔名種類)
> 2. Img_Rename -> 將圖片名稱改成Faster RCNN之圖片名稱形式 (ex:000001.jpg)
> 3. test_draw_box -> 自行設定要測試的圖片及其對應的Bbox座標，執行後可用來確認Bbox位置是否正確

-------------------------
## logs  
### 2020/10/7  
> 將json file之資料輸出成Faster RCNN之roidb資料，並完成其Data Augmentation  

### 2020/10/8  
<<<<<<< HEAD
<<<<<<< HEAD
> 改將json file資料輸出成xml檔案，並修改Data Augmentation為輸出xml檔  

### 2020/10/11
> 將Data Augmentation修改，改為先做水平+垂直翻轉後再將全部imgae做Rotate


=======
> 改將json file資料輸出成xml檔案，並修改Data Augmentation為輸出xml檔
>>>>>>> 2559c4d9115ad2fe4a57493c2901c6ec8a5a5e9b
