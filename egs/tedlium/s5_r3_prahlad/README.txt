Full Name: Prahlad Koratamaddi
UNI: pk2743

Date: 18th December, 2022

Project Title: 
Exploring State-Of-The-Art Language Models for Speech Recognition in TED-LIUM 3

Project Summary: 
Language modeling plays a crucial role in the task of Automatic Speech Recognition (ASR). In various state-of-the-art ASR systems, language models (LMs) are used along with acoustic and pronunciation models in calculating the conditional probability over a sequence of words, given the audio input features. Numerous comparative studies have shown that an ASR system utilizing a language model has a lower Word Error Rate (WER) than one without it. Additionally, lowering the overall perplexity of the language model appears to also reduce the WER of an ASR system using the language model. In this project, I studied the problem of improving the TED-LIUM (release 3) ASR models by enhancing the underlying language model. By implementing three advanced language models including efficient convolutional-based and state-of-the-art attention-based models, I perform decoding using various rescoring strategies and conduct experiments on the effect of language models in ASR systems. I also run my experiments on the LibriSpeech test dataset, allowing for evaluation of the effects on independent data. Overall, my findings suggest that key factors such as the evaluation and training datasets, fine-tuning, perplexity and architecture significantly effect the performance of the Language Model and consequently the ASR system. 

Tools Used:
-> Python 3.6.9:
    - PyTorch 1.10.2
    - Transformers 4.18.0
    

Executing/Testing the Code:
    -> REQUIRED - Activating the virtual environment: 
    
    -> Directory of the Project: tedlium/s5_r3_prahlad/

    -> Main script: run_pk2743.sh
        - Command to execute for decoding: ./run_pk2743.sh
        
        - Default configuration: Decodes using the Transformer LM on TED-LIUM 3 test dataset with N-Best Lists Rescoring.
        
        - Other configurations can be defined as options. Examples include: 
        1) ./run_pk2743.sh --model pytorch_transformer --decode_on_tedlium true --nbest false
        same as default but instead of N-best rescoring it does lattice rescoring.
        
        2) ./run_pk2743.sh --model transformer_xl --decode_on_tedlium false --nbest false
        decodes using Transformer XL LM on LibriSpeech test dataset with lattice rescoring.
        
        3) ./run_pk2743.sh --model gcnnlm --decode_on_tedlium true --nbest true
        decodes using the Gated Convolutional NNLM 



