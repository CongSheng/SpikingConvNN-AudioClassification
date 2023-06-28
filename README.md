#   Sparsity through Spiking Convolutional Neural Network (SCNN) for Audio Classification at the Edge
Cong Sheng Leow, Wang Ling Goh, Yuan Gao\
Published in IEEE International Symposium on Circuits and Systems 2023 (ISCAS 2023).\
![Vector for image](https://github.com/CongSheng/Research/blob/4e846fa81fe949b0dbbe98b45d2c19e26beb617b/figures/Vector(JPEG).jpg)

Abstract: Convolutional neural networks (CNNs) have shown to be effective for audio classification. However, deep CNNs can be computationally heavy and unsuitable for edge intelligence as embedded devices are generally constrained by memory and energy requirements. Spiking neural networks (SNNs) offer potential as energy-efficient networks but typically underperform typical deep neural networks in accuracy. This paper proposes a spiking convolutional neural network (SCNN) that exhibits excellent accuracy of above 98 % on a multi-class audio classification task. Accuracy remains high with weight quantization to INT8- precision. Additionally, this paper examines the role of neuron pagit rameters in co-optimizing activation sparsity and accuracy.
![Model A Architecture](https://github.com/CongSheng/Research/blob/8e32179f69676ef81428b0c1c8b36818afa801b5/figures/ModelA.jpg)
*Fig 1: Model A Network (SCNN) inspired by the original CNN used for FSDD.*
![Model B Architecture](https://github.com/CongSheng/Research/blob/8e32179f69676ef81428b0c1c8b36818afa801b5/figures/ModelB.jpg)
*Fig 2: Model B Network (SCNN) with leaner architecture than model A.*
![Training Loss](https://github.com/CongSheng/Research/blob/8e32179f69676ef81428b0c1c8b36818afa801b5/figures/Output%20Plots/accLoss.png)
*Training loss plot against training epochs for Model B.*
![Confusion Matrix](https://github.com/CongSheng/Research/blob/a319623006914158a6865fbc23d4aa5f2301cbe0/figures/Output%20Plots/ConfusionMatrix.png)
*Confusion matrix for model B.*

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Architecture</th>
            <th>No. of time steps</th>
            <th>Test Accuracy</th>
            <th>Quantized Test Accuracy</th>
            <th>No. of Parameters</th>
            <th>FLOPS</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align: center">Control (CNN)</td>
            <td style="text-align: center">6C3-P2-16C3-128FC-64FC-10FC</td>
            <td style="text-align: center">NA</td>
            <td style="text-align: center">97.3 %</td>
            <td style="text-align: center">-</td>
            <td style="text-align: center">0.837 M</td>
            <td style="text-align: center">3.326M</td>
        </tr>
        <tr>
            <td style="text-align: center" rowspan=2>A</td>
            <td style="text-align: center" rowspan=2>6C3-16C3-128FC-64FC-10FC</td>
            <td style="text-align: center">10</td>
            <td style="text-align: center">98.7 &#xB1; 0.57 %</td>
            <td style="text-align: center">74.1 &#xB1; 9.22 %</td>
            <td style="text-align: center" rowspan=2>1.616 M</td>
            <td style="text-align: center">281.000 M</td>
        </tr>
        <tr>
            <td style="text-align: center">1</td>
            <td style="text-align: center">95.4 &#xB1; 3.24 %</td>
            <td style="text-align: center">61.6 &#xB1; 12.30 %</td>
            <td style="text-align: center">28.085 M</td>
        </tr>
        <tr>
            <td style="text-align: center" rowspan=2> B</td>
            <td style="text-align: center" rowspan=2>6C3-128FC-10FC</td>
            <td style="text-align: center">10</td>
            <td style="text-align: center">98.4 &#xB1; 0.35 %</td>
            <td style="text-align: center">97.0 &#xB1; 0.77 %</td>
            <td style="text-align: center" rowspan=2>0.702 M</td>
            <td style="text-align: center">88.930 M</td>
        </tr>
        <tr>
            <td style="text-align: center">1</td>
            <td style="text-align: center">93.1 &#xB1; 0.77 %</td>
            <td style="text-align: center">91.3 &#xB1; 2.40 %</td>
            <td style="text-align: center">8.893 M </td>
        </tr>
    </tbody>
</table>
Table showing the results for the model. Benchmark table available in original 
paper.

##  Table of Content
1. [Structure](#Structure)
2. [Dependencies](#Dependencies)
3. [Usage](#Usage)
4. [Scripts](#Scripts)
5. [Future Work](#Future_Work)
6. [Acknowledgements](#Acknowledgements)

## Structure
```
Folder PATH listing for volume Windows
Volume serial number is 945B-7E82
C:.
│   .gitignore
│   automatedSearch.py
│   feature_exploration.py
│   iii_quantize.py
│   ii_spikingTest.py
│   iv_layerAnalysis.py
│   i_print_dict.py
│   main.py
│   manualSearch.py
│   README.md
│   requirements.txt
│   scriptRun.py
││   
├───checkpoints    
|    
├───datasets
│   │   customDataset.py
│   │   mfcc_dataset.py
│   │   __init__.py
│           
├───Expt
│   │   expt.log
│   │   exptPlot.log
│   │   exptProfile.log
|
├───figures
├───free-spoken-digit-dataset-v1.0.8
│   └───FSDD
│       │   .gitignore
│       │   metadata.py
│       │   pip_requirements.txt
│       │   README.md
│       │   __init__.py
│       │   
│       ├───acquire_data
│       │       say_numbers_prompt.py
│       │       split_and_label_numbers.py
│       │       
│       ├───recordings
│       │    
│       └───utils
│               fsdd.py
│               spectogramer.py
│               trimmer.py
│               __init__.py
│               
├───hyperTuning
│   │   parameterScriptRun.log
│   │   profileScriptRun.log
│   │   results.log
│   │   resultsScriptRun.log
│   │   tuningPara.log
│   │       
│   ├───confuseMatrix
│   │       
│   └───Train
│           
├───logs
├───models
│   │   AlexCNN.py
│   │   CustomCNN.py
│   │   LeNet.py
│   │   test.py
│   │   train.py
│           
├───transformedData
│               
├───utils
│   │   audioProcessing.py
│   │   loggingFn.py
│   │   plotFigure.py
│   │   spikingNeuron.py
│   │   __init__.py
```

## Dependencies
It recommended for this to be run on [docker](https://www.docker.com/) or a 
[virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Click on the above links to get started. 
```
connda install pip
conda install -file requirements.txt
```
or `$cond acreate --name <environment_name> -- file requirements.txt` to create
a new enivronment with the packages ready.

## Usage
The whole repository can be divided in several levels:
1. Input
2. Output
3. Utility and Self-defined Packages
4. Python Scripts

### Input
You should first download the dataset to the home directory. This project
focuses on the free spoken digit dataset (FSDD) which can be downloaded via
https://github.com/Jakobovski/free-spoken-digit-dataset.

### Output
There are different types of files this project works with. Some of these
include: .png (figures), .log(log files), .pt (model checkpoints and transformed
features). These are mainly stored at their individual folders, but the 
directory can be specified at the scripts which produces them (typically at the
top to configure log path or figure path).

### Utility and Self-defined Packages
Utility packages are written mainly for re-use in the scripts for purposes such
as signal processing, logging, or plotting figures. These are located in the 
`utils` folder. On the other hand, the models and scripts to train or test the 
networks are located inside the `models` folder.

### Python Scripts
Ideally, most modification should only be done on the scripts within the home
directory. These scripts offer configuration options at the command-line level
through `argparse` for simple modifications. Realistically, modifications can 
also be done at the script level, especially at directory configuration, change
in sizes of plots, and etc. More explanations will be illustrated at the
following section on the scripts.

## Scripts
The scripts can be segmented into numbered scripts and non-numbered scripts. 
Numbered scripts are scripts which were mainly used to better understand the
functions or logic behind the tools and options available. On the other hand, 
non-numbered scripts are more stand-alone. Instructions on the arguments are
written within the `argparse`'s `-help` section. \

**Numbered Scripts**\
`i_print_dict.py`: Used to visualize dictionaries, such as the checkpoints.\
`ii_spikingTest.py`: Used to visualize spiking neurons through printed plot.\
`iii_quantize.py`: Used for Quantized-Aware-Training (QAT).\
`iv_layerAnalysis`: Used to visualize and analysis layer's weights and output.\

**Non-numbered Scripts**\
`automatedSearch`: Validation/Search through neuron parameters using *Tune*.\
![Automated Search](https://github.com/CongSheng/Research/blob/a319623006914158a6865fbc23d4aa5f2301cbe0/figures/Search/combined_search.png)
*Random and Bayesian search for highest accuracy across neuron parameters for both Model A and Model B: Beta, No. of time steps, and Threshold voltage.*
`feature_exploration.py`: Used to explore different features on a single audio.\
`main.py`: Main script to train and evaluate different models at different configurations.\
`manualSearch.py`: Manually sweep along individual neuron parameters to examine
the accuracy and sparasity.\
![Manual Search for Model A](https://github.com/CongSheng/Research/blob/a319623006914158a6865fbc23d4aa5f2301cbe0/figures/Search/HyperStudy%20Subplot%20Aonly.png)
*Manual Search for Model A - Varying single parameter while keeping the rest constant.*
![Manual Search for Model B](https://github.com/CongSheng/Research/blob/a319623006914158a6865fbc23d4aa5f2301cbe0/figures/Search/HyperStudy%20Subplot%20Bonly.png)
*Manual Search for Model B - Varying single parameter while keeping the rest constant.*
`scriptRun.py`: Similar to `main.py`, but without `argparse` and requires 
modification directly on the script.

## Future Works
1. Implemention and test of hardware implementation.
2. Multi-objective optimization (Accuracy, sparsity and hardware resources).
3. Online optimization.

## Acknowledgements
This work is built upon the great works of others, some of them are listed below.
However, for the full list of references, please do refer to the bibliography
of the original publication. Do also cite this work if you think it is helpful
in your work!

Other useful works:\
snnTorch: https://github.com/jeshraghian/snntorch/tree/master \
Brevitas: https://github.com/Xilinx/brevitas \
Flop Counter: https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md \
Ray (Tune): https://github.com/ray-project/ray
