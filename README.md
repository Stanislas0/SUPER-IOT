# SUPER-IOT
The official implementation of the following [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9123371): 

> *Towards Secure and Efﬁcient Deep Learning Inference in Dependable IoT Systems*
>
> Han Qiu, Qinkai Zheng, Tianwei Zhang, Meikang Qiu, Gerard Memmi, Jialiang Lu
>
> *Abstract*: The rapid development of Deep Learning (DL) enables resource-constrained systems and devices (e.g., Internet of Things) to perform sophisticated Artiﬁcial Intelligence (AI) applications. However, AI models like Deep Neural Networks (DNNs) are known to be vulnerable to Adversarial Examples (AEs). Past works on defending against AEs require heavy computations in the model training or inference processes, making them impractical to be applied in IoT systems. In this paper, we propose a novel method, S UPER -I OT, to enhance the security and efﬁciency of AI applications in distributed IoT systems. Speciﬁcally, S UPER -I OT utilizes a pixel drop operation to eliminate adversarial perturbations from the input and reduce network transmission throughput. Then it adopts a sparse signal recovery method to reconstruct the dropped pixels and waveletbased denoising method to reduce the artiﬁcial noise. S UPER I OT is a lightweight method with negligible computation cost to IoT devices and little impact on the DNN model performance. Extensive evaluations show that it can outperform three existing AE defensive solutions against most of the AE attacks with better transmission efﬁciency.

If you have any question, please raise an issue or contact ```qinkai.zheng1028@gmail.com```. 

## Requirements

* cleverhans==3.0.1
* opencv_python==4.1.2.30
* Keras==2.2.4
* scikit_image==0.16.2
* numpy==1.19.1
* tensorflow==1.14.0

## Citation

``````
@article{qiu2020towards,
  title={Towards secure and efficient deep learning inference in dependable IoT systems},
  author={Qiu, Han and Zheng, Qinkai and Zhang, Tianwei and Qiu, Meikang and Memmi, Gerard and Lu, Jialiang},
  journal={IEEE Internet of Things Journal},
  year={2020},
  publisher={IEEE}
}
``````

