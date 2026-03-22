# Research Report: Plant Disease Classification with Izhikevich SNNs

## Experiment Context

This experiment explored the performance of hybrid Convolutional Neural Network-Spiking Neural Network (CNN-SNN) architectures for plant disease classification using the PlantVillage dataset. Five different Izhikevich neuron types were compared in the SNN layers to understand their impact on model accuracy and training dynamics.

- **Dataset**: PlantVillage Dataset (raw/color subset)
- **Number of Classes**: 38
- **Image Size**: 128x128
- **Total Dataset Samples**: 54,305
- **Subset Fraction Used**: 35%
- **Training Samples**: 13,304
- **Validation Samples**: 2,850
- **Test Samples**: 2,852
- **SNN Timesteps (T)**: 8
- **Batch Size**: 32
- **Training Epochs**: 25
- **Learning Rate (LR)**: 0.0003
- **Weight Decay**: 0.0001
- **Early Stopping Patience**: 8

## Key Findings

The models were trained and evaluated across five distinct Izhikevich neuron types. The best performing model on the test set achieved the following results:

- **Best Performing Neuron Type**: Intrinsic Bursting (IB) (IB)
- **Test Accuracy**: 98.42%
- **Best Validation Accuracy**: 98.77% (achieved at Epoch 24)

## Neuroscience Analysis of the Best Performing Neuron Type: Intrinsic Bursting (IB)

### Izhikevich Parameters

- **a (recovery speed)**: 0.02
- **b (sub-threshold sensitivity)**: 0.2
- **c (after-spike reset voltage)**: -55.0
- **d (after-spike recovery jump)**: 4.0

### Brain Region

Neocortical Layer V — Thick-tufted pyramidal neurons (Primary Motor Cortex, Somatosensory Cortex, Visual Cortex L5b). Project to thalamus, brainstem, spinal cord.

### Biological Mechanism

IB neurons fire an initial 2–5 spike burst ('salience alarm') followed by repetitive single spikes. c=−55 mV gives an intermediate reset voltage between RS (−65) and CH (−50). Large apical dendrites integrate broad spatial regions; the burst onset reliably signals the appearance of a new visual stimulus.

### Image Classification Verdict (Biological Analogy)

STRONG. Excellent salience detection: the burst onset fires strongly when a distinctive feature (disease spot, necrotic patch) appears in the CNN feature map. Slightly weaker than CH for diffuse symptoms like early-stage yellowing where the burst advantage diminishes. Competes closely with RS overall. Predicted rank: 3rd.

### Additional Insights into Winning Performance


1. **Salience onset detection**
   The initial burst acts as a 'salience alarm': when a new distinctive CNN feature (disease spot, necrotic patch) appears, IB neurons fire a strong burst that dominates the readout and draws the classifier's attention to it.


2. **Intermediate reset voltage**
   c=−55 balances RS and CH: enough depolarisation for fast re-fire on salient features, but enough hyperpolarisation to suppress noise. This intermediate point generalises well across the heterogeneous visual appearances of 38 classes.


3. **Large spatial integration**
   Layer-V apical dendrites integrate broad receptive fields, complementing the CNN's local convolutional features with global spatial context.
