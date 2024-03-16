# HCRVD
  Deep-learning-based vulnerability detection methods extract syntax and semantic information of the program through the code itself or logical structures to detect vulnerabilities, where code representation greatly affects the detection effect. In this paper, we propose a Hierarchical Code Representation learning based Vulnerability Detection system (HCRVD), which uses a combination of Concrete Syntax Tree (CST) and Program Dependence Graph (PDG) of the source code. 
HCRVD consists of five phases: Data preparation, Function PDG extraction, Statement CST parsing, Vector transformation, TGGA Construction and Classification.

  Data preparation: Extracting functions from source code, labelling whether functions contain vulnerabilities, and normalizing them.

  Function PDG extraction: A PDG is generated from each normalized function, and subgraphs of the PDG are extracted as a means of data augmentation [28] based on possible vulnerability-sensitive words for that function.

  Statement CST parsing: The code statements of each node extracted from the PDG are parsed into CSTs individually and depth-first traversal is performed from each CST to extract the information of the node, generating token sequences and statement CST tree-structure data.

  Dynamic Vector transformation: A dictionary is built and CP data are transformed into vectors. Gated Recurrent Units (GRU) are used to dynamically build the network structure based on the maximum statement CST and the maximum number of PDG nodes (or thresholds) on batch to enable batch processing. 

  TGGA Construction and Classification: Building a TGGA network and using GRUs to transfer information in different levels according to data structures. The function representation learned by TGGA is used for function granularity vulnerability detection.
# Implementation Steps
## Step 1: Data preparation
```
python 1\ preparation.py -i ./data/source
```
## Step 2: generate pdg with joern
```
python 2\ joern_pdg.py -i ./data/source/Vul -o ./data/bins/Vul -t parse
python 2\ joern_pdg.py -i ./data/source/No-Vul -o ./data/bins/No-Vul -t parse
python 2\ joern_pdg.py -i ./data/bins/Vul -o ./data/dots/Vul -t export -r pdg
python 2\ joern_pdg.py -i ./data/bins/No-Vul -o ./data/dots/No-Vul -t export -r pdg
```
## Step 3: generate pdg dataset
```
python 3\ gen_pdg_dataset.py -i ./data/dots/ -o ./data/pdgs/
```
## Step 4: parse CST
```
python 4\ CST\ parsing.py -i ./data/pdgs/
```
## Step 5: build and train TGGA
```
python 5\ train.py -i ./data/pdgs/ -o ./output/
```
# Quick Start
```
python 5\ train.py -i ./data/pdgs/ -o ./output/
```
