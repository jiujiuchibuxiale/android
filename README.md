# 实验四
## 一、下载
### 1.直接访问链接并下载其中的zip包，解压至工作目录中
## 二、对项目文件进行更新
### 1.升级项目文件的grandle
### 2.修改build.grandle
## 三、向应用中添加TensorFlow Lite
### 1.选择file，然后New->Other->TensorFlow Lite Model
### 2.选择已经下载的自定义的训练模型，并完成模型导入，系统将自动下载模型的依赖包并将依赖项添加至模块的build.gradle文件；并生成摘要信息
### 3.检查代码中的TODO项
### 4.添加代码

<p>&emsp;&emsp;1.定位“start”模块MainActivity.kt文件的TODO 1，添加初始化训练模型的代码</p>
private class ImageAnalyzer(ctx: Context, private val listener: RecognitionListener) :
        ImageAnalysis.Analyzer {

  ...
  // TODO 1: Add class variable TensorFlow Lite Model
  private val flowerModel = FlowerModel.newInstance(ctx)

  ...
}
<p>&emsp;&emsp;2.在CameraX的analyze方法内部，需要将摄像头的输入ImageProxy转化为Bitmap对象，并进一步转化为TensorImage 对象</p>
override fun analyze(imageProxy: ImageProxy) {
  ...
  // TODO 2: Convert Image to Bitmap then to TensorImage
  val tfImage = TensorImage.fromBitmap(toBitmap(imageProxy))
  ...
}
<p>&emsp;&emsp;3.对图像进行处理并生成结果，主要包含下述操作：
&emsp;&emsp;&emsp;按照属性score对识别结果按照概率从高到低排序
&emsp;&emsp;&emsp;列出最高k种可能的结果，k的结果由常量MAX_RESULT_DISPLAY定义</p>
override fun analyze(imageProxy: ImageProxy) {
  ...
  // TODO 3: Process the image using the trained model, sort and pick out the top results
  val outputs = flowerModel.process(tfImage)
      .probabilityAsCategoryList.apply {
          sortByDescending { it.score } // Sort with highest confidence first
      }.take(MAX_RESULT_DISPLAY) // take the top results

  ...
}
<p>&emsp;&emsp;4.将识别的结果加入数据对象Recognition 中，包含label和score两个元素。后续将用于RecyclerView的数据显示</p>
override fun analyze(imageProxy: ImageProxy) {
  ...
  // TODO 4: Converting the top probability items into a list of recognitions
  for (output in outputs) {
      items.add(Recognition(output.label, output.score))
  }
  ...
}
<p>&emsp;&emsp;5.将原先用于虚拟显示识别结果的代码注释掉或者删除</p>
// START - Placeholder code at the start of the codelab. Comment this block of code out.
for (i in 0..MAX_RESULT_DISPLAY-1){
    items.add(Recognition("Fake label $i", Random.nextFloat()))
}
// END - Placeholder code at the start of the codelab. Comment this block of code out.
