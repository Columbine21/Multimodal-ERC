## THUIAR Multimodal Emotion Recognization in Conversation

### Content

- Introduction.
- Baseline Classifications And Implement (Public).
- Updates And Timeline.

### Introduction

**Emotion Recognization in Conversation(ERC)** is one of the most popular subtasks of utterance-level dialogue

 understanding. The task is to predict the emotion labels (happy, sad, neutral, angry, excited, frustrated, 

disgust, and fear) of the constituent utterance, where each utterance is uttered by one specific speaker.

### Classification of existing methods

All existing model can be classified into 4 classes:

1. Context Modeling Methods;
2. Methods Transfered from **Multimodal Sentiment Analysis(MSA)** ;
3.  Memory Network Methods;
4. Graph Neural Network Methods;

#### Context Modeling Methods

This kind of Methods focus on Modeling **CONTROLLING VARIABLES IN CONVERSATIONS** (proposed in

 the survey "Emotion Recognition in Conversation: Research Challenges, Datasets, and Recent Advances")

  and **build complex context control module**.

Typical Methods：

1. bc-lstm: Context-Dependent Sentiment Analysis in User-Generated Videos [2017 02]
2. DialogueRnn: DialogueRNN: An Attentive RNN for Emotion Detection in Conversations [2019 05]

Our Code Base: **NOW SUPPORT bc-lstm & dialogueRnn on IEMOCAP & MELD** 

​																	TO BE CONTINUE …

#### Methods Transfered from MSA 

This kind of Methods focus on how to use multimodal information on different multimodal datasets.

 But failed to take different speaker & speaker's history utterance into account. They are usually used

 to verify proposed multimodal methods having the ability of modeling context & speakers' info.

Typical Methods:

1. TFN :Tensor Fusion Network for Multimodal Sentiment Analysis [2017 07]
2. MFN: Memory Fusion Network for Multi-view Sequential Learning [2018 02]

Our Code Base: **NOW DEVELOPING** 

​																	TO BE CONTINUE …

#### Memory Network Methods

In the literature, memory networks have been successfully applied in many areas, including 

question-answering, machine translation, speech recognition, and others. Inspired by its capabilities of

context modeling, Multi-hops  Memory Network is proposed in ERC Tasks, and verified to be effective.

Typical Methods: 

1. CMN: CMN-Interactive conversational memory network for multimodal emotion detection [2018 06]
2. ICON: ICON Interactive conversational memory network for multimodal emotion detection [2018 xx]

Our Code Base: **NOW DEVELOPING** 

​																	TO BE CONTINUE …

#### Graph Neural Network Methods

This kind of Methods use different Graph Neural Network to make improvements in ERC tasks. 

Typical Methods:

1. Congcn: Modeling both Context-and Speaker-Sensitive Dependence for Emotion Detect-

   ion in Multi-speaker Conversations [2019 01]

2. DialogueGCN: Dialoguegcn- A graph convolutional neural network for emotion recognition

    in conversation [2019 08]

Our Code Base: **NOW DEVELOPING** 

​																	TO BE CONTINUE …

