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
  
| Dataset | Method       | Rouge-L | Bleu  | Meteor |
|---------|--------------|---------|-------|--------|
| LCSTS   | Transformer  | 36.4    | 16.38 | 35.59  |
|         | OurModel     | **37.35** | **17.39** | **36.19** |
| CSL     | Transformer  | 55.27   | 34.50 | 57.02  |
|         | OurModel     | **55.82** | **35.00** | **57.40** |
