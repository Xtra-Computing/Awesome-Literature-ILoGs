# Awesome Literature for [Imbalanced Learning on Graphs](xxx.xxx.xxx) (ILoGs)
This repository showcases a curated collection of research literature on imbalanced learning on graphs. We've categorized this literature according to the taxonomies of problems and techniques detailed in our survey paper, titled [A Survey of Imbalanced Learning on Graphs: Problems, Techniques, and Future Directions](xxx.xxx.xxx). In this repository, we primarily arrange the literature based on our problem taxonomy for clarity. For a deeper understanding of this rapidly evolving and challenging field, we encourage readers to consult our survey.

For our taxonomy of Problems, we classify the literature based on class imbalance and structure imbalance, both stemming from imbalanced input. We further distill this into more specific categories: node-, edge-, and graph-level imbalance, offering a comprehensive understanding of graph imbalance.

For a more comprehensive overview of imbalanced learning on various data, please refer to Github Repository [Awesome-Imbalanced-Learning](https://github.com/yanliang3612/awesome-imbalanced-learning-on-geometric-and-graphs).

## Outline

- [1. Class Imbalance](https://github.com/shuaiOKshuai/ILoGs#1-class-imbalance)
  - [1.1 Node-Level Class Imbalance](https://github.com/shuaiOKshuai/ILoGs#11-node-level-class-imbalance)
    - [1.1.1 Imbalanced Node CLassification](https://github.com/shuaiOKshuai/ILoGs#111-imbalanced-node-classification)
    - [1.1.2 Node-Level Anomaly Detection](https://github.com/shuaiOKshuai/ILoGs#112-node-level-anomaly-detection)
    - [1.1.3 Few-Shot Node Classification](https://github.com/shuaiOKshuai/ILoGs#113-few-shot-node-classification)
    - [1.1.4 Zero-Shot Node Classification](https://github.com/shuaiOKshuai/ILoGs#114-zero-shot-node-classification)
  - [1.2 Edge-Level Class Imbalance](https://github.com/shuaiOKshuai/ILoGs#12-edge-level-class-imbalance)
    - [1.2.1 Few-Shot Link Prediction](https://github.com/shuaiOKshuai/ILoGs#121-few-shot-link-prediction)
    - [1.2.2 Edge-Level Anomaly Detection](https://github.com/shuaiOKshuai/ILoGs#122-edge-level-anomaly-detection)
  - [1.3 Graph-Level Class Imbalance](https://github.com/shuaiOKshuai/ILoGs#13-graph-level-class-imbalance)
    - [1.3.1 Imbalanced Graph Classification](https://github.com/shuaiOKshuai/ILoGs#131-imbalanced-graph-classification)
    - [1.3.2 Graph_Level Anomaly Detection](https://github.com/shuaiOKshuai/ILoGs#132-graph_level-anomaly-detection)
    - [1.3.3 Few-Shot Graph Classification](https://github.com/shuaiOKshuai/ILoGs#133-few-shot-graph-classification)
- [2. Structure Imbalance](https://github.com/shuaiOKshuai/ILoGs#2-structure-imbalance)
  - [2.1 Node-Level Structure Imbalance](https://github.com/shuaiOKshuai/ILoGs#21-node-level-structure-imbalance)
    - []()
    - []()
    - []()
  - []()
    - []()
    - []()
    - []()
  - []()
    - []()
    - []()



## 1. Class Imbalance

### 1.1 Node-Level Class Imbalance

#### 1.1.1 Imbalanced Node CLassification

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| DR-GCN  | Multi-Class Imbalanced Graph Convolutional Network Learning | IJCAI 2020  |  [PDF](https://www.ijcai.org/proceedings/2020/398)  | [TensorFlow](https://github.com/codeshareabc/DRGCN)  |
| DPGNN  | Distance-wise Prototypical Graph Neural Network for Imbalanced Node Classification  | arXiv 2021  | [PDF](https://arxiv.org/abs/2110.12035)  | [PyTorch](https://github.com/YuWVandy/DPGNN)  |
| GraphSMOTE  | GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks  | WSDM 2021  | [PDF](https://arxiv.org/abs/2103.08826)  | [PyTorch](https://github.com/TianxiangZhao/GraphSmote)  |
| ImGAGN  | ImGAGN: Imbalanced Network Embedding via Generative Adversarial Graph Networks  | KDD 2021  | [PDF](https://arxiv.org/abs/2106.02817)  | [PyTorch](https://github.com/Leo-Q-316/ImGAGN)  |
| TAM  | TAM: Topology-Aware Margin Loss for Class-Imbalanced Node Classification  | ICML 2022  | [PDF](https://proceedings.mlr.press/v162/song22a/song22a.pdf)  | [PyTorch](https://github.com/Jaeyun-Song/TAM)  |
| LTE4G  | LTE4G: Long-Tail Experts for Graph Neural Networks  | CIKM 2022  | [PDF](https://arxiv.org/abs/2208.10205)  | [PyTorch](https://github.com/SukwonYun/LTE4G)  |
| GraphMixup  | GraphMixup: Improving Class-Imbalanced Node Classification on Graphs by Self-supervised Context Prediction  | ECML-PKDD 2022  | [PDF](https://arxiv.org/abs/2106.11133)  | [PyTorch](https://github.com/LirongWu/GraphMixup)  |
| GraphENS  | GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification   | ICLR 2022  | [PDF](https://openreview.net/forum?id=MXEl7i-iru)  | [PyTorch](https://github.com/JoonHyung-Park/GraphENS)  |
| GraphSANN  | Imbalanced Node Classification Beyond Homophilic Assumption  | IJCAI 2023  | [PDF](https://arxiv.org/abs/2304.14635)  | 	[N/A]  |
| ALLIE  | ALLIE: Active Learning on Large-scale Imbalanced Graphs  | WWW 2022  | [PDF](https://dl.acm.org/doi/10.1145/3485447.3512229)  | [N/A]  |



#### 1.1.2 Node-Level Anomaly Detection

#### 1.1.3 Few-Shot Node Classification

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Meta-GNN  | Meta-GNN: On Few-shot Node Classification in Graph Meta-learning  | CIKM 2019  | [PDF](https://arxiv.org/abs/1905.09718)  | [PyTorch](https://github.com/ChengtaiCao/Meta-GNN)  |
| AMM-GNN  | Graph Few-shot Learning with Attribute Matching  | CIKM 2020  | [PDF](https://www.public.asu.edu/~kding9/pdf/CIKM2020_AMM.pdf)  | [N/A]  |
| TRGM  | Task-level Relations Modelling for Graph Meta-learning  | ICDM 2022  | [PDF](https://ieeexplore.ieee.org/document/10027781)  | [N/A]  |
| RALE  | Relative and Absolute Location Embedding for Few-Shot Node Classification on Graph  | AAAI 2021  | [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/16551)  | [TensorFlow](https://github.com/shuaiOKshuai/RALE)  |
| GFL  | Graph Few-shot Learning via Knowledge Transfer  | AAAI 2020  | [PDF](https://arxiv.org/abs/1910.03053)  | [PyTorch](https://github.com/huaxiuyao/GFL)  |
| GPN  | Graph Prototypical Networks for Few-shot Learning on Attributed Networks  | Content  | [PDF](https://arxiv.org/abs/2006.12739)  | [PyTorch](https://github.com/kaize0409/GPN_Graph-Few-shot)  |
| MuL-GRN  | MuL-GRN: Multi-Level Graph Relation Network for Few-Shot Node Classification  | TKDE 2022  | [PDF](https://ieeexplore.ieee.org/document/9779997)  | [N/A]  |
| ST-GFSL  | Spatio-Temporal Graph Few-Shot Learning with Cross-City Knowledge Transfer  | KDD 2022  | [PDF](https://arxiv.org/abs/2205.13947)  | [PyTorch](https://github.com/RobinLu1209/ST-GFSL)  |
| G-Meta  | Graph Meta Learning via Local Subgraphs  | NeurIPS 2020  | [PDF](https://arxiv.org/abs/2006.07889)  | [PyTorch](https://github.com/mims-harvard/G-Meta)  |
| TENT  | Task-Adaptive Few-shot Node Classification  | KDD 2022  | [PDF](https://arxiv.org/abs/2206.11972)  | [PyTorch](https://github.com/SongW-SW/TENT)  |
| Meta-GPS  | Few-shot Node Classification on Attributed Networks with Graph Meta-learning  | SIGIR 2022  | [PDF](https://dl.acm.org/doi/abs/10.1145/3477495.3531978)  | [N/A]  |
| IA-FSNC  | Information Augmentation for Few-shot Node Classification  | IJCAI 2022  | [PDF](https://www.ijcai.org/proceedings/2022/500)  | 	[N/A]  |
| SGCL  | Supervised Graph Contrastive Learning for Few-shot Node Classification  | ECML-PKDD 2022  | [PDF](https://arxiv.org/abs/2203.15936)  | [N/A]  |
| TLP  | Transductive Linear Probing: A Novel Framework for Few-Shot Node Classification  | LoG 2022  | [PDF](https://arxiv.org/abs/2212.05606)  | [PyTorch](https://github.com/Zhen-Tan-dmml/TLP-FSNC)  |
| GraphPrompt  | GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks  | WWW 2023  | [PDF](https://arxiv.org/abs/2302.08043)  | [PyTorch](https://github.com/Starlien95/GraphPrompt)  |
| Stager  | Generalized Few-Shot Node Classification  | ICDM 2022  | [PDF](https://ieeexplore.ieee.org/document/10027718)  | [PyTorch](https://github.com/pricexu/STAGER)  |
| HAG-Meta  | Graph Few-shot Class-incremental Learning  | WSDM 2022  | [PDF](https://arxiv.org/abs/2112.12819)  | [PyTorch](https://github.com/Zhen-Tan-dmml/GFCIL)  |
| Geometer  | Geometer: Graph Few-Shot Class-Incremental Learning via Prototype Representation  | KDD 2022  | [PDF](https://arxiv.org/abs/2205.13954)  | [PyTorch](https://github.com/RobinLu1209/Geometer)  |
| MetaTNE  | Node Classification on Graphs with Few-Shot Novel Labels via Meta Transformed Network Embedding  | NeurIPS 2020  | [PDF](https://arxiv.org/abs/2007.02914)  | [PyTorch](https://github.com/llan-ml/MetaTNE)  |
| X-FNC  | Few-shot Node Classification with Extremely Weak Supervision  | WSDM 2023  | [PDF](https://arxiv.org/abs/2301.02708)  | [PyTorch](https://github.com/SongW-SW/X-FNC)  |
| CrossHG-Meta  | Few-shot Heterogeneous Graph Learning via Cross-domain Knowledge Transfer  | KDD 2022  | [PDF](https://dl.acm.org/doi/abs/10.1145/3534678.3539431)  | [N/A]  |
| HG-Meta  | HG-Meta: Graph Meta-learning over Heterogeneous Graphs  | SDM 2022  | [PDF](https://epubs.siam.org/doi/10.1137/1.9781611977172.45)  | [N/A]  |



#### 1.1.4 Zero-Shot Node Classification

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| RSDNE  | RSDNE: Exploring Relaxed Similarity and Dissimilarity from Completely-Imbalanced Labels for Network Embedding  | AAAI 2018  | [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/11242)  | [MATLAB](https://github.com/zhengwang100/RSDNE)  |
| RECT  | Network Embedding with Completely-imbalanced Labels  | TKDE 2020  | [PDF](https://arxiv.org/abs/2007.03545)  | [PyTorch](https://github.com/zhengwang100/RECT)  |
| RECT  | Expanding Semantic Knowledge for Zero-shot Graph Embedding  | DASFAA 2021  | [PDF](https://arxiv.org/abs/2103.12491)  | [N/A]  |
| DGPN  | Zero-shot Node Classification with Decomposed Graph Prototype Network  | KDD 2021  | [PDF](https://arxiv.org/abs/2106.08022)  | [PyTorch](https://github.com/zhengwang100/dgpn)  |
| DBiGCN  | Dual Bidirectional Graph Convolutional Networks for Zero-shot Node Classification  | KDD 2022  | [PDF](http://www.lamda.nju.edu.cn/conf/mla22/paper/yq-KDD2022.pdf)  | [PyTorch](https://github.com/warmerspring/DBiGCN)  |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |


### 1.2 Edge-Level Class Imbalance

#### 1.2.1 Few-Shot Link Prediction

#### 1.2.2 Edge-Level Anomaly Detection

### 1.3 Graph-Level Class Imbalance

#### 1.3.1 Imbalanced Graph Classification

#### 1.3.2 Graph_Level Anomaly Detection

#### 1.3.3 Few-Shot Graph Classification

## 2. Structure Imbalance

### 2.1 Node-Level Structure Imbalance

#### 2.1.1 Imbalanced Node Degrees

#### 2.1.2 Node Topology Imbalance

#### 2.1.3 Long-Tail Entity Embedding

### 2.2 Edge-Level Structure Imbalance 

#### 2.2.1 Few-Shot Relation Classification

#### 2.2.2 Zero-Shot Relation Classification

#### 2.2.3 Few-Shot Reasoning on KGs

### 2.3 Graph-Level Structure Imbalance

#### 2.3.1 Imbalanced Graph Sizes

#### 2.3.2 Imbalanced Topology Groups

# Acknowledge
This page is contributed and maintained by Zemin Liu (zeminliu@nus.edu.sg), Yuan Li (xxxx), and Nan Chen (xxxx).
