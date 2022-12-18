Full Name: Prahlad Koratamaddi
UNI: pk2743

Date: 18th December, 2022

Project Title: 
Exploring State-Of-The-Art Language Models for Speech Recognition in TED-LIUM 3

Project Summary: 
Language modeling plays a crucial role in the task of Automatic Speech Recognition (ASR). In various state-of-the-art ASR systems, language models (LMs) are used along with acoustic and pronunciation models in calculating the conditional probability over a sequence of words, given the audio input features. Numerous comparative studies have shown that an ASR system utilizing a language model has a lower Word Error Rate (WER) than one without it. Additionally, lowering the overall perplexity of the language model appears to also reduce the WER of an ASR system using the language model. In this project, I studied the problem of improving the TED-LIUM (release 3) ASR models by enhancing the underlying language model. By implementing three advanced language models including efficient convolutional-based and state-of-the-art attention-based models, I perform decoding using various rescoring strategies and conduct experiments on the effect of language models in ASR systems. I also run my experiments on the LibriSpeech test dataset, allowing for evaluation of the effects on independent data. Overall, my findings suggest that key factors such as the evaluation and training datasets, fine-tuning, perplexity and architecture significantly effect the performance of the Language Model and consequently the ASR system. 

Tools Used:
-> Notable: 
    - Python 3.6.9:
        -- PyTorch 1.10.2
        -- Transformers 4.18.0
-> Others: 
    - as mentioned in requirements.txt file    

Executing/Testing the Code:    
    -> Directory of the Project: tedlium/s5_r3_prahlad/

    -> REQUIRED - Activating the virtual environment: 
        -- please navigate to the project directory
        -- next, run this command: source pk2743_venv/bin/activate 
        -- this would activate the directory

    -> Main script: run_pk2743.sh
        - Command to execute for decoding: ./run_pk2743.sh

        - Expected Output: 
            Stage 19 start
            Decoding with the Transformer NNLM.............
            steps/pytorchnn/lmrescore_nbest_pytorchnn.sh --cmd run.pl --max-jobs-run 1 --model-type Transformer --embedding_dim 768 --hidden_dim 768 --nlayers 8 --nhead 8 --weight 0.7 --oov-symbol '<UNK>' --stage 7 data/lang_chain exp/pytorch_transformer/model.pt data/pytorchnn/words.txt data/test_hires exp/chain_cleaned_1d/tdnn1d_sp/decode_test exp/chain_cleaned_1d/tdnn1d_sp/decode_test_pytorch_transformer_nbest
            steps/pytorchnn/lmrescore_nbest_pytorchnn.sh: reconstructing total LM+graph scores including interpolation of neural LM and old LM scores.
            steps/pytorchnn/lmrescore_nbest_pytorchnn.sh: reconstructing archives back into lattices.
            scoring...
            exp/chain_cleaned_1d/tdnn1d_sp/decode_test_pytorch_transformer_nbest/wer_10
            %WER 7.32 [ 2014 / 27500, 304 ins, 610 del, 1100 sub ]
            %SER 63.72 [ 736 / 1155 ]
            exp/chain_cleaned_1d/tdnn1d_sp/decode_test_pytorch_transformer_nbest/wer_11
            %WER 7.39 [ 2032 / 27500, 300 ins, 643 del, 1089 sub ]
            %SER 63.64 [ 735 / 1155 ]
            exp/chain_cleaned_1d/tdnn1d_sp/decode_test_pytorch_transformer_nbest/wer_12
            %WER 7.47 [ 2054 / 27500, 305 ins, 658 del, 1091 sub ]
            %SER 63.90 [ 738 / 1155 ]
            exp/chain_cleaned_1d/tdnn1d_sp/decode_test_pytorch_transformer_nbest/wer_13
            %WER 7.56 [ 2080 / 27500, 306 ins, 678 del, 1096 sub ]
            %SER 64.16 [ 741 / 1155 ]
            exp/chain_cleaned_1d/tdnn1d_sp/decode_test_pytorch_transformer_nbest/wer_14
            %WER 7.58 [ 2084 / 27500, 301 ins, 690 del, 1093 sub ]
            %SER 64.50 [ 745 / 1155 ]
            exp/chain_cleaned_1d/tdnn1d_sp/decode_test_pytorch_transformer_nbest/wer_15
            %WER 7.63 [ 2099 / 27500, 304 ins, 698 del, 1097 sub ]
            %SER 65.19 [ 753 / 1155 ]
            exp/chain_cleaned_1d/tdnn1d_sp/decode_test_pytorch_transformer_nbest/wer_16
            %WER 7.71 [ 2121 / 27500, 306 ins, 714 del, 1101 sub ]
            %SER 65.28 [ 754 / 1155 ]
            exp/chain_cleaned_1d/tdnn1d_sp/decode_test_pytorch_transformer_nbest/wer_17
            %WER 7.72 [ 2123 / 27500, 301 ins, 724 del, 1098 sub ]
            %SER 65.63 [ 758 / 1155 ]
            exp/chain_cleaned_1d/tdnn1d_sp/decode_test_pytorch_transformer_nbest/wer_7
            %WER 7.32 [ 2014 / 27500, 335 ins, 508 del, 1171 sub ]
            %SER 63.03 [ 728 / 1155 ]
            exp/chain_cleaned_1d/tdnn1d_sp/decode_test_pytorch_transformer_nbest/wer_8
            %WER 7.20 [ 1979 / 27500, 319 ins, 533 del, 1127 sub ]
            %SER 62.68 [ 724 / 1155 ]
            exp/chain_cleaned_1d/tdnn1d_sp/decode_test_pytorch_transformer_nbest/wer_9
            %WER 7.24 [ 1991 / 27500, 310 ins, 571 del, 1110 sub ]
            %SER 63.29 [ 731 / 1155 ]

        - Default configuration: 
            -- Decodes using the Transformer LM on TED-LIUM 3 test dataset with N-Best Lists Rescoring.
        
        - Other configurations can be defined as options:
            -- Models available: pytorch_transformer, transformer_xl and gcnnlm
            -- Datasets available: TED-LIUM 3 and LibriSpeech test datasets
            -- Rescoring strategies: N-Best and Lattice Rescoring

        Examples include: 
        1) ./run_pk2743.sh --model pytorch_transformer --decode_on_tedlium true --nbest false
        same as default but instead of N-best rescoring it does lattice rescoring.
        
        2) ./run_pk2743.sh --model transformer_xl --decode_on_tedlium false --nbest false
        decodes using Transformer XL LM on LibriSpeech test dataset with lattice rescoring.
        
        3) ./run_pk2743.sh --model gcnnlm --decode_on_tedlium true --nbest true
        decodes using the Gated Convolutional NNLM on TED-LIUM test dataset with N-best rescoring

    -> Other useful files: 
        - RESULTS file: 
            -- consists of the best decoding scores for all experiments (all model + data + rescoring strategy combinations) conducted on the data.            
            -- can be used to check the results mentioned on the paper

        - RESULTS_<model>_LM files:
            -- consist of the individual model decoding results across all data + rescoring strategy combinations.
            -- can be used to verify the execution outputs for each of the configurations

        




