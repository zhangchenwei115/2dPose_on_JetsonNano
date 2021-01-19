2dPose_on_JetsonNano
此文件主要用于记录Jetson Nano上部署openpose，需要注意一点是pip install pycocotools会报错，类似 egg info check什么的，是因为matplotlib安装失败，此时需要 sudo apt-get install python-matplotlib的方式安装，在尝试pip3 install pycocotools即可成功。  
下面这主要解释一下这个repo里做了什么。  
# Nano上运行原版demo
运行方式是运行文件originaldemo.py。因为nano上用的csi摄像头，故需要改一下摄像头的运行方式，所以对于demo.py代码有所改动。运行方式为
```bash
python demo.py --checkpoint-path <path_to>/checkpoint_iter_370000.pth
```
不需要加上--video 0

# 用torch2trt加速
torch2trt在博客里面说过怎么安装，之后运行converttensorrt.py代码，稍等片刻即可转换成功。然后直接运行demoRT.py.即可加速。需要注意的的一点是，这块可能是我没有搞清楚的，所以我采用比较笨的办法。就是在torch2trt转换的时候需要定下输入的大小，原作者定义的为（1，3，256，344），因为摄像头的输入不知道怎么修改，输入只能为1280，720.所以我在demoRT.py里的VideoReader函数里面修改了视频输入大小，即： img = cv2.resize(img,(int(344),int(256)))。  
## 添加FPS
首先要在外面定义ftp_time = 0  
然后在视频的循环里添加  
```bash
#FPS
global fps_time
now = time.time()
cv2.putText(img,
            "FPS: %f" % (1.0 / (now - fps_time)),
            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2)


cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
fps_time = time.time()
```
即可显示帧率
