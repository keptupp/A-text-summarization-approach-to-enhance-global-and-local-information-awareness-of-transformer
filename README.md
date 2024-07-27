# A Text Summarization Approach to Enhance Global and Local Information Awareness of Transformer
Paper experiment source code

This project comes from one of our papers **Text-summaries-of-global-and-local-information-awareness**

## Abstract
In the field of abstract text summarization, architectures based on encoder-decoder frameworks are widely applied to sequence-to-sequence generation tasks and can effectively handle sequences of unlimited length.
Subsequently, the transformer model use a global attention mechanism, allowing encodings at different distances to mutually interact, greatly enhancing the model's contextual awareness.
However, this context-awareness is global, requiring the model to additionally learn to extract different levels of information to increase understanding.
We improve the structure of the model to introduce prior knowledge so that it can learn from the global and local information and enhance the model's understanding ability.
This paper proposes global information-aware encoding and local information-aware encoding,
which enhance the understanding of documents from coarse-grained and fine-grained perspectives respectively.
Global encoding adds an extra feature to the encoder stage and performs attention with the document,
generating a global summary encoding of the entire document to guide the generation of the summary content.
Local encoding is to perform local convolution on the features extracted by the encoder,
use prior knowledge to extract local features of the document and enable the model to quickly extract local detail information.
Experiments show that the improved model proposed in this paper has higher rouge scores than the baseline model on the LCSTS and CSL datasets,
and also has advantages over some mainstream models. The generated summaries are more accurate and informative.

## Code Introduce
The code is divided into 3 independent parts. mycode is the first part, training inference code on LCSTS and CSL data sets for the original transformer model, mycode02 is training inference code on LCSTS for the improved model, and mycode03 is training and reasoning code on CSL for the improved model. And mycode03 also includes ablation experiments.  
  
The training framework of each part is basically the same. Run main.py to start training or inference, and set the model structure, data set path and other parameters in config.py.

## Model Performance



The following table compares the proposed model with the transformer model in various indicators.
| Dataset | Method       | Rouge-L | Bleu  | Meteor |
|---------|--------------|---------|-------|--------|
| LCSTS   | Transformer  | 36.4    | 16.38 | 35.59  |
|         | OurModel     | **37.35** | **17.39** | **36.19** |
| CSL     | Transformer  | 55.27   | 34.50 | 57.02  |
|         | OurModel     | **55.82** | **35.00** | **57.40** |




  
The following is a comparison between our proposed model and some mainstream models on LCSTS.
| Models       | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------------|---------|---------|---------|
| TD-NHG\cite{li2022news} | 31.28   | 12.68   | 28.31   |
| WeLM\cite{su2022welm}   | 32.23   | -       | -       |
| TI-C-NHG\cite{li2023topic} | 34.26 | 16.74   | 32.03   |
| GP\_Step\_0.3            | 36.24 | 22.56   | 34.36   |
| Transformer              | 40.3  | 27.0    | 36.4    |
| OurModel                 | **41.0** | **28.1** | **37.3** |
  


The following is a comparison between our proposed model and some mainstream models on the CSL dataset.
| Models                   | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------------------------|---------|---------|---------|
| Original T5 250\cite{wang2023t5} | 56.45   | 45.01   | 53.96   |
| PEGASUS\cite{zhang2020pegasus}   | -       | -       | 55.2    |
| BART\cite{lewis-etal-2020-bart}  | -       | -       | 49.9    |
| CSL-T5\cite{li2022csl}          | -       | -       | 52.1    |
| LSTM-seq2seq\cite{wang2023t5}    | 46.48   | 30.48   | 41.8    |
| Transformer                    | 60.43   | 46.57   | 55.27   |
| OurModel                       | **60.94** | **47.16** | **55.82** |



The following is an ablation experiment to explore the effects of our two proposed coding structures on the model, conducted on the CSL dataset.
| Models       | Parmas   | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------------|----------|---------|---------|---------|
| Transformer  | 76.61M   | 60.43   | 46.57   | 55.27   |
| +global      | 76.66M   | 60.44   | 46.66   | 55.38   |
| +local       | 86.37M   | 60.36   | 46.63   | 55.19   |
| OurModel     | 96.88M   | **60.94** | **47.16** | **55.82** |
