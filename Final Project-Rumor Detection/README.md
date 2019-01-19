# Final Project: Rumor Detection
This Project is the Final Project of [DATA130006](http://www.sdspeople.fudan.edu.cn/zywei/DATA130006/index.html)

## Remark
* This Project is a group work implemented by Pingxuan Huang, Guozhen She, Cenguang Zhang. Codes are available, if you want to utilize them, however, please indicate the source;
* Dataset is available at [here](http://www.sdspeople.fudan.edu.cn/zywei/DATA130006/final-project/index.html). But, before successfully downloading the dataset, **please contact Dc.Zhongyu Wei for admission**
* If you have any question about the codes or the dataset, please don't hesitate to contact *1336076538@qq.com* for help

## Introduction
### Requirement
* This project aims to determine if a post is rumor or not given its original content and all the comments and reposts generated;
* Some related papers are provided as reference (you can obtain them at the [*References*](#reference) part). Students can either choose one of them to implement;
* Two datasets from Weibo and Twitter can be found at [here](http://www.sdspeople.fudan.edu.cn/zywei/DATA130006/final-project/index.html)

### Our work
Based on all the 3 papers provided by professor, we implement not one but all the 3 alternative models:
* **Dynamic SeriesTime Structure (DSTS)/SVM/LR**
* **LSTM/Bi-LSTM model**
* **Propagation Tree Kernel (PTK) model.**

Please check *Report* for the detail information about the model and our experiment. By the way, we also provide the code to crawl Twitter Data from the internet.

## <span id="reference"> References </span>
### References provided by professor
\[1\] Wei Gao Kam-Fai Wong Jing Ma. 2017. Detect rumors in microblog posts using propagation structure via kernel learning. ACL pages 708–717

\[2\] Wei Gao Prasenjit Mitra Sejeong Kwon Bernard J. Jansen Kam-Fai Wong Meeyoung Cha. Wu S. Yang Jing Ma and K. Q. Zhu. 2016. Detecting rumors from microblogs with recurrent neural networks. IJCAI pages 3818–3824

\[3\] Wei Gao Zhongyu Wei Yueming Lu Jing Ma and KamFai Wong. 2015. Detect rumors using time series of social context information on microblogging websites. CIKM

### Additional references
\[4\] M. Mendoza C. Castillo and B. Poblete. 2011. Information credibility on twitter. In Proceedings of WWW page 675-684

\[5\] S. Yang K. Wu and K. Q. Zhu. 2015. False rumors detection on sina weibo by propagation structures. In Proceedings of ICDE page 675-684
