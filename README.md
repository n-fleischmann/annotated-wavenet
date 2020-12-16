# Annotated Wavenet

The purpose of this repository and the corresponding paper, annotated_wavenet.pdf, is to annotate and explain
DeepMind’s WaveNet architecture. The module is divided into several submodules that collate data, define the 
models parts and then train and generate. This implentation is based on the version available here:
https://github.com/ryujaehun/wavenet
    
The model is capable of identifying and imitating the rhythmic features of the seed file and not much more. With
more computational power and a larger receptive field, it could be expanded to creating samples like those found
in DeepMind's blog post here:
https://deepmind.com/blog/article/wavenet-generative-model-raw-audio
