# PrepareDataByJson
this project is use to prepare the Faster RCNN Dataset by labelme's json file.  
(You need to prepare your own data)

## 主要實現功能
> 將labelme標註後所輸出的json file進行處理，輸出成Faster RCNN之Dataset形式  
> 並且將其輸出5個可視化資料(範例):  
 ![image]https://github.com/minwei1997/PrepareDataByJson/blob/main/sample.PNG

-------------------------
## 使用方法
> 將labelme標註之json檔放進 ".\data\js_data\js_file" 路徑，接著執行json_extract.py即可進行處理
> Faster RCNN之Dataset存至 ".\data\training_pickle"
> 5個可視化資料存至 ".\data\js_data\js_output"
