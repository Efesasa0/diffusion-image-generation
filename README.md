# DDPM From Scratch

Implementation of the Image Generation, Diffusion model adapted from the [paper](https://arxiv.org/abs/2006.11239). To test the capability to the most simple case, the model has been trained on the custom "Sprites" dataset from the [DeepLearning.ai](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) course.

## Getting Started

```bash
    git clone https://github.com/Efesasa0/diffusion-image-generation.git
    cd diffusion-image-generation
    pip install -r requirements.txt
    python train.py
```

### References

* deeplearing.ai course and dataset: [How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/)
* [CIFAR-10](https://www.kaggle.com/c/cifar-10)
* MNIST
* [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

#### Additional References

* [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
* [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
