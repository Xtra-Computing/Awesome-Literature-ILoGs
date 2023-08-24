# Awesome Literature on [Imbalanced Learning on Graphs](xxx.xxx.xxx) (ILoGs)
This repository showcases a curated collection of research literature on imbalanced learning on graphs. We have categorized this literature according to the taxonomies of **Problems** and **Techniques** detailed in our survey paper, titled [A Survey of Imbalanced Learning on Graphs: Problems, Techniques, and Future Directions](xxx.xxx.xxx). In this repository, we primarily arrange the literature based on our **Problem** taxonomy for clarity. For a deeper understanding of this rapidly evolving and challenging field, we encourage readers to consult our survey.

For our taxonomy of **Problems**, we classify the literature based on **class imbalance** and **structure imbalance**, both stemming from imbalanced input. We further distill this into more specific categories: node-, edge-, and graph-level imbalance, offering a comprehensive understanding of graph imbalance.

For a more comprehensive overview of imbalanced learning on various data, please refer to Github Repository [Awesome-Imbalanced-Learning](https://github.com/yanliang3612/awesome-imbalanced-learning-on-geometric-and-graphs).

Please note that the order of papers within each category may not strictly adhere to chronological sequence; instead, it generally aligns with the structure presented in our survey.

# Outline

The outline corresponds to the taxonomy of Problems in our [survey paper](xxx.xxx.xxx).

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
    - [1.3.2 Graph-Level Anomaly Detection](https://github.com/shuaiOKshuai/ILoGs#132-graph_level-anomaly-detection)
    - [1.3.3 Few-Shot Graph Classification](https://github.com/shuaiOKshuai/ILoGs#133-few-shot-graph-classification)
- [2. Structure Imbalance](https://github.com/shuaiOKshuai/ILoGs#2-structure-imbalance)
  - [2.1 Node-Level Structure Imbalance](https://github.com/shuaiOKshuai/ILoGs#21-node-level-structure-imbalance)
    - [2.1.1 Imbalanced Node Degrees](https://github.com/shuaiOKshuai/ILoGs#211-imbalanced-node-degrees)
    - [2.1.2 Node Topology Imbalance](https://github.com/shuaiOKshuai/ILoGs#212-node-topology-imbalance)
    - [2.1.3 Long-Tail Entity Embedding on KGs](https://github.com/shuaiOKshuai/ILoGs#213-long-tail-entity-embedding-on-kgs)
  - [2.2 Edge-Level Structure Imbalance](https://github.com/shuaiOKshuai/ILoGs#22-edge-level-structure-imbalance)
    - [2.2.1 Few-Shot Relation Classification](https://github.com/shuaiOKshuai/ILoGs#221-few-shot-relation-classification)
    - [2.2.2 Zero-Shot Relation Classification](https://github.com/shuaiOKshuai/ILoGs#222-zero-shot-relation-classification)
    - [2.2.3 Few-Shot Reasoning on KGs](https://github.com/shuaiOKshuai/ILoGs#223-few-shot-reasoning-on-kgs)
  - [2.3 Graph-Level Structure Imbalance](https://github.com/shuaiOKshuai/ILoGs#23-graph-level-structure-imbalance)
    - [2.3.1 Imbalanced Graph Sizes](https://github.com/shuaiOKshuai/ILoGs#231-imbalanced-graph-sizes)
    - [2.3.2 Imbalanced Topology Groups](https://github.com/shuaiOKshuai/ILoGs#232-imbalanced-topology-groups)

# Literature

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
| ALLIE  | ALLIE: Active Learning on Large-scale Imbalanced Graphs  | WWW 2022  | [PDF](https://dl.acm.org/doi/10.1145/3485447.3512229)  | [N/A]  |
| GraphSANN  | Imbalanced Node Classification Beyond Homophilic Assumption  | IJCAI 2023  | [PDF](https://arxiv.org/abs/2304.14635)  | 	[N/A]  |




#### 1.1.2 Node-Level Anomaly Detection

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |


#### 1.1.3 Few-Shot Node Classification

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Meta-GNN  | Meta-GNN: On Few-shot Node Classification in Graph Meta-learning  | CIKM 2019  | [PDF](https://arxiv.org/abs/1905.09718)  | [PyTorch](https://github.com/ChengtaiCao/Meta-GNN)  |
| AMM-GNN  | Graph Few-shot Learning with Attribute Matching  | CIKM 2020  | [PDF](https://www.public.asu.edu/~kding9/pdf/CIKM2020_AMM.pdf)  | [N/A]  |
| GFL  | Graph Few-shot Learning via Knowledge Transfer  | AAAI 2020  | [PDF](https://arxiv.org/abs/1910.03053)  | [PyTorch](https://github.com/huaxiuyao/GFL)  |
| GPN  | Graph Prototypical Networks for Few-shot Learning on Attributed Networks  | CIKM 2020  | [PDF](https://arxiv.org/abs/2006.12739)  | [PyTorch](https://github.com/kaize0409/GPN_Graph-Few-shot)  |
| G-Meta  | Graph Meta Learning via Local Subgraphs  | NeurIPS 2020  | [PDF](https://arxiv.org/abs/2006.07889)  | [PyTorch](https://github.com/mims-harvard/G-Meta)  |
| MetaTNE  | Node Classification on Graphs with Few-Shot Novel Labels via Meta Transformed Network Embedding  | NeurIPS 2020  | [PDF](https://arxiv.org/abs/2007.02914)  | [PyTorch](https://github.com/llan-ml/MetaTNE)  |
| RALE  | Relative and Absolute Location Embedding for Few-Shot Node Classification on Graph  | AAAI 2021  | [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/16551)  | [TensorFlow](https://github.com/shuaiOKshuai/RALE)  |
| MuL-GRN  | MuL-GRN: Multi-Level Graph Relation Network for Few-Shot Node Classification  | TKDE 2022  | [PDF](https://ieeexplore.ieee.org/document/9779997)  | [N/A]  |
| ST-GFSL  | Spatio-Temporal Graph Few-Shot Learning with Cross-City Knowledge Transfer  | KDD 2022  | [PDF](https://arxiv.org/abs/2205.13947)  | [PyTorch](https://github.com/RobinLu1209/ST-GFSL)  |
| TENT  | Task-Adaptive Few-shot Node Classification  | KDD 2022  | [PDF](https://arxiv.org/abs/2206.11972)  | [PyTorch](https://github.com/SongW-SW/TENT)  |
| Meta-GPS  | Few-shot Node Classification on Attributed Networks with Graph Meta-learning  | SIGIR 2022  | [PDF](https://dl.acm.org/doi/abs/10.1145/3477495.3531978)  | [N/A]  |
| IA-FSNC  | Information Augmentation for Few-shot Node Classification  | IJCAI 2022  | [PDF](https://www.ijcai.org/proceedings/2022/500)  | 	[N/A]  |
| SGCL  | Supervised Graph Contrastive Learning for Few-shot Node Classification  | ECML-PKDD 2022  | [PDF](https://arxiv.org/abs/2203.15936)  | [N/A]  |
| TLP  | Transductive Linear Probing: A Novel Framework for Few-Shot Node Classification  | LoG 2022  | [PDF](https://arxiv.org/abs/2212.05606)  | [PyTorch](https://github.com/Zhen-Tan-dmml/TLP-FSNC)  |
| Stager  | Generalized Few-Shot Node Classification  | ICDM 2022  | [PDF](https://ieeexplore.ieee.org/document/10027718)  | [PyTorch](https://github.com/pricexu/STAGER)  |
| HAG-Meta  | Graph Few-shot Class-incremental Learning  | WSDM 2022  | [PDF](https://arxiv.org/abs/2112.12819)  | [PyTorch](https://github.com/Zhen-Tan-dmml/GFCIL)  |
| Geometer  | Geometer: Graph Few-Shot Class-Incremental Learning via Prototype Representation  | KDD 2022  | [PDF](https://arxiv.org/abs/2205.13954)  | [PyTorch](https://github.com/RobinLu1209/Geometer)  |
| CrossHG-Meta  | Few-shot Heterogeneous Graph Learning via Cross-domain Knowledge Transfer  | KDD 2022  | [PDF](https://dl.acm.org/doi/abs/10.1145/3534678.3539431)  | [N/A]  |
| HG-Meta  | HG-Meta: Graph Meta-learning over Heterogeneous Graphs  | SDM 2022  | [PDF](https://epubs.siam.org/doi/10.1137/1.9781611977172.45)  | [N/A]  |
| TRGM  | Task-level Relations Modelling for Graph Meta-learning  | ICDM 2022  | [PDF](https://ieeexplore.ieee.org/document/10027781)  | [N/A]  |
| X-FNC  | Few-shot Node Classification with Extremely Weak Supervision  | WSDM 2023  | [PDF](https://arxiv.org/abs/2301.02708)  | [PyTorch](https://github.com/SongW-SW/X-FNC)  |
| GraphPrompt  | GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks  | WWW 2023  | [PDF](https://arxiv.org/abs/2302.08043)  | [PyTorch](https://github.com/Starlien95/GraphPrompt)  |



#### 1.1.4 Zero-Shot Node Classification

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| RSDNE  | RSDNE: Exploring Relaxed Similarity and Dissimilarity from Completely-Imbalanced Labels for Network Embedding  | AAAI 2018  | [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/11242)  | [MATLAB](https://github.com/zhengwang100/RSDNE)  |
| RECT  | Network Embedding with Completely-imbalanced Labels  | TKDE 2020  | [PDF](https://arxiv.org/abs/2007.03545)  | [PyTorch](https://github.com/zhengwang100/RECT)  |
| RECT  | Expanding Semantic Knowledge for Zero-shot Graph Embedding  | DASFAA 2021  | [PDF](https://arxiv.org/abs/2103.12491)  | [N/A]  |
| DGPN  | Zero-shot Node Classification with Decomposed Graph Prototype Network  | KDD 2021  | [PDF](https://arxiv.org/abs/2106.08022)  | [PyTorch](https://github.com/zhengwang100/dgpn)  |
| DBiGCN  | Dual Bidirectional Graph Convolutional Networks for Zero-shot Node Classification  | KDD 2022  | [PDF](http://www.lamda.nju.edu.cn/conf/mla22/paper/yq-KDD2022.pdf)  | [PyTorch](https://github.com/warmerspring/DBiGCN)  |



### 1.2 Edge-Level Class Imbalance

#### 1.2.1 Few-Shot Link Prediction

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| EA-GAT  | Few-Shot Link Prediction for Event-Based Social Networks via Meta-learning  | DASFAA 2023  | [PDF](https://link.springer.com/chapter/10.1007/978-3-031-30675-4_3)  | [N/A]  |

#### 1.2.2 Edge-Level Anomaly Detection

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |

### 1.3 Graph-Level Class Imbalance

#### 1.3.1 Imbalanced Graph Classification

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| $\text{G}^2\text{GNN}$  | Imbalanced Graph Classification via Graph-of-Graph Neural Networks | CIKM 2022  | [PDF](https://arxiv.org/abs/2112.00238)  | [PyTorch](https://github.com/YuWVandy/G2GNN)  |


#### 1.3.2 Graph-Level Anomaly Detection

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |

#### 1.3.3 Few-Shot Graph Classification

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| AS-MAML  | Adaptive-Step Graph Meta-Learner for Few-Shot Graph Classification  | CIKM 2020  | [PDF](https://arxiv.org/abs/2003.08246)  | [PyTorch](https://github.com/NingMa-AI/AS-MAML)  |
|   | Few-Shot Learning on Graphs via Super-Classes based on Graph Spectral Measures  | ICLR 2020  | [PDF](https://arxiv.org/abs/2002.12815)  | [PyTorch](https://github.com/chauhanjatin10/GraphsFewShot)  |
| PAR  | Property-Aware Relation Networks for Few-Shot Molecular Property Prediction  | NeurIPS 2021  | [PDF](https://arxiv.org/abs/2107.07994)  | [PyTorch](https://github.com/tata1661/PAR-NeurIPS21)  |
| Meta-MGNN  | Few-Shot Graph Learning for Molecular Property Prediction  | WWW 2021  | [PDF](https://arxiv.org/abs/2102.07916)  | [PyTorch](https://github.com/zhichunguo/Meta-MGNN)  |
| FAITH  | FAITH: Few-Shot Graph Classification with Hierarchical Task Graphs  | IJCAI-ECAI 2022  | [PDF](https://arxiv.org/abs/2205.02435)  | [PyTorch](https://github.com/SongW-SW/FAITH)  |
|   | Metric Based Few-Shot Graph Classification  | LoG 2022  | [PDF](https://arxiv.org/abs/2206.03695)  | [PyTorch](https://github.com/crisostomi/metric-few-shot-graph)  |
|   | Cross-Domain Few-Shot Graph Classification  | AAAI 2022  | [PDF](https://arxiv.org/abs/2201.08265)  | [N/A]  |
| Temp-GFSM  | Meta-Learned Metrics over Multi-Evolution Temporal Graphs  | KDD 2022  | [PDF](https://dl.acm.org/doi/10.1145/3534678.3539313)  | [PyTorch](https://github.com/LiriFang/Temp-GFSM)  |
|   | Graph Neural Network Expressivity and Meta-Learning for Molecular Property Regression  | LoG 2022  | [PDF](https://arxiv.org/abs/2209.13410)  | [N/A]  |
| ADKF-IFT  | Meta-learning Adaptive Deep Kernel Gaussian Processes for Molecular Property Prediction  | ICLR 2023  | [PDF](https://arxiv.org/abs/2205.02708)  | [PyTorch](https://github.com/Wenlin-Chen/ADKF-IFT)  |
| MTA  | Meta-Learning with Motif-based Task Augmentation for Few-Shot Molecular Property Prediction  | SDM 2023  | [PDF](https://epubs.siam.org/doi/10.1137/1.9781611977653.ch91)  | [N/A]  |


## 2. Structure Imbalance

### 2.1 Node-Level Structure Imbalance

#### 2.1.1 Imbalanced Node Degrees

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Demo-Net  | DEMO-Net: Degree-specific Graph Neural Networks for Node and Graph Classification  | KDD 2019  | [PDF](https://arxiv.org/pdf/1906.02319.pdf)  | [TensorFlow](https://github.com/jwu4sml/DEMO-Net)  |
| SL-DSGCN  | Investigating and Mitigating Degree-Related Biases in Graph Convolutional Networks  | CIKM 2020  | [PDF](https://arxiv.org/abs/2006.15643)  | [N/A]  |
| DHGAT  | A Dual Heterogeneous Graph Attention Network to Improve Long-Tail Performance for Shop Search in E-Commerce  | KDD 2020  | [PDF](http://shichuan.org/hin/topic/Similarity%20Measure/KDD2020.A%20Dual%20Heterogeneous%20Graph%20Attention%20Network%20to%20Improve%20Long-Tail%20Performance%20for%20Shop%20Search%20in%20E-Commerce.pdf)  | [N/A]  |
| meta-tail2vec  | Towards Locality-Aware Meta-Learning of Tail Node Embeddings on Networks  | CIKM 2020  | [PDF](https://zemin-liu.github.io/papers/CIKM-20-towards-locality-aware-meta-learning-of-tail-node-embeddings-on-network.pdf)  | [TensorFlow](https://github.com/smufang/meta-tail2vec)  |
| Tail-GNN  | Tail-GNN: Tail-Node Graph Neural Networks  | KDD 2021  | [PDF](https://zemin-liu.github.io/papers/Tail-GNN-KDD-21.pdf)  | [PyTorch](https://github.com/shuaiOKshuai/Tail-GNN)  |
|   | Pre-Training Graph Neural Networks for Cold-Start Users and Items Representation  | WSDM 2021  | [PDF](https://arxiv.org/abs/2012.07064)  | [TensorFlow](https://github.com/jerryhao66/Pretrain-Recsys)  |
| Residual2Vec  | Residual2Vec: Debiasing graph embedding with random graphs  | NeurIPS 2021  | [PDF](https://arxiv.org/abs/2110.07654)  | [PyTorch](https://github.com/skojaku/residual2vec)  |
| CenGCN  | CenGCN: Centralized Convolutional Networks with Vertex Imbalance for Scale-Free Graphs  | TKDE 2022  | [PDF](https://arxiv.org/abs/2202.07826)  | [N/A]  |
| MetaDyGNN  | Few-shot Link Prediction in Dynamic Networks  | WSDM 2022  | [PDF](http://www.shichuan.org/doc/120.pdf)  | [PyTorch](https://github.com/BUPT-GAMMA/MetaDyGNN)  |
| Cold Brew  | Cold Brew: Distilling Graph Node Representations with Incomplete or Missing Neighborhoods  | ICLR 2022  | [PDF](https://arxiv.org/abs/2111.04840)  | [PyTorch](https://github.com/amazon-science/gnn-tail-generalization)  |
| BLADE  | BLADE: Biased Neighborhood Sampling based Graph Neural Network for Directed Graphs  | WSDM 2023  | [PDF](https://dl.acm.org/doi/abs/10.1145/3539597.3570430)  | [N/A]  |


#### 2.1.2 Node Topology Imbalance

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ReNode  | Topology-Imbalance Learning for Semi-Supervised Node Classification  | NeurIPS 2021  | [PDF](https://arxiv.org/abs/2110.04099)  | [PyTorch](https://github.com/victorchen96/ReNode)  |
| PASTEL  | Position-aware Structure Learning for Graph Topology-imbalance by Relieving Under-reaching and Over-squashing  | CIKM 2022  | [PDF](https://arxiv.org/abs/2208.08302)  | [PyTorch](https://github.com/RingBDStack/PASTEL)  |
| HyperIMBA  | Hyperbolic Geometric Graph Representation Learning for Hierarchy-imbalance Node Classification  | WWW 2023  | [PDF](https://arxiv.org/abs/2304.05059)  | [PyTorch](https://github.com/RingBDStack/HyperIMBA)  |

#### 2.1.3 Long-Tail Entity Embedding on KGs

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| GEN  | Learning to Extrapolate Knowledge: Transductive Few-shot Out-of-Graph Link Prediction  | NeurIPS 2020  | [PDF](https://arxiv.org/abs/2006.06648)  | [PyTorch](https://github.com/JinheonBaek/GEN)  |
| DAT  | Degree-Aware Alignment for Entities in Tail  | SIGIR 2020  | [PDF](https://arxiv.org/abs/2005.12132)  | [PyTorch](https://github.com/DexterZeng/DAT)  |
| OKELE  | Open Knowledge Enrichment for Long-tail Entities  | WWW 2020  | [PDF](https://arxiv.org/abs/2002.06397)  | [TensorFlow](https://github.com/nju-websoft/OKELE/)  |
| MaKEr  | Meta-Learning Based Knowledge Extrapolation for Knowledge Graphs in the Federated Setting  | 	IJCAI 2022  | [PDF](https://arxiv.org/abs/2205.04692)  | [PyTorch](https://github.com/zjukg/MaKEr)  |
| MorsE  | Meta-Knowledge Transfer for Inductive Knowledge Graph Embedding  | SIGIR 2022  | [PDF](https://arxiv.org/abs/2110.14170)  | [PyTorch](https://github.com/zjukg/MorsE)  |
| MTKGE  | Meta-Learning Based Knowledge Extrapolation for Temporal Knowledge Graph  | WWW 2023  | [PDF](https://arxiv.org/abs/2302.05640)  | [N/A]  |
| KG-Mixup  | Toward Degree Bias in Embedding-Based Knowledge Graph Completion  | WWW 2023  | [PDF](https://arxiv.org/abs/2302.05044)  | [PyTorch](https://github.com/HarryShomer/KG-Mixup)  |



### 2.2 Edge-Level Structure Imbalance 

#### 2.2.1 Few-Shot Relation Classification

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Gmatching  | One-Shot Relational Learning for Knowledge Graphs  | EMNLP 2018  | [PDF](https://arxiv.org/abs/1808.09040)  | [PyTorch](https://github.com/xwhan/One-shot-Relational-Learning)  |
| Proto-HATT  | Hybrid Attention-Based Prototypical Networks for Noisy Few-Shot Relation Classification  | AAAI 2019  | [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/4604)  | [PyTorch](https://github.com/thunlp/HATT-Proto)  |
| MetaR  | Meta Relational Learning for Few-Shot Link Prediction in Knowledge Graphs  | EMNLP 2019  | [PDF](https://arxiv.org/abs/1909.01515)  | [PyTorch](https://github.com/AnselCmy/MetaR)  |
|   | Tackling Long-Tailed Relations and Uncommon Entities in Knowledge Graph Completion  | EMNLP-IJCNLP 2019  | [PDF](https://aclanthology.org/D19-1024/)  | [PyTorch](https://github.com/ZihaoWang/Few-shot-KGC)  |
| FSRL  | Few-Shot Knowledge Graph Completion  | AAAI 2020  | [PDF](https://arxiv.org/abs/1911.11298)  | [PyTorch](https://github.com/chuxuzhang/AAAI2020_FSRL)  |
| FAAN  | Adaptive Attentional Network for Few-Shot Knowledge Graph Completion  | EMNLP 2020  | [PDF](https://aclanthology.org/2020.emnlp-main.131/)  | [PyTorch](https://github.com/JiaweiSheng/FAAN)  |
| Mick  | MICK: A Meta-Learning Framework for Few-shot Relation Classification with Small Training Data  | CIKM 2020  | [PDF](https://arxiv.org/abs/2004.14164)  | [PyTorch](https://github.com/XiaoqingGeng/MICK)  |
| Neural Snowball  | Neural Snowball for Few-Shot Relation Learning  | AAAI 2020  | [PDF](https://arxiv.org/abs/1908.11007)  | [PyTorch](https://github.com/thunlp/Neural-Snowball) |
| REFORM  | REFORM: Error-Aware Few-Shot Knowledge Graph Completion  | CIKM 2021  | [PDF](https://chenannie45.github.io/CIKM21a.pdf)  | [PyTorch](https://github.com/SongW-SW/REFORM)  |
| P-INT  | P-INT: A Path-based Interaction Model for Few-shot Knowledge Graph Completion  | EMNLP 2021  | [PDF](https://aclanthology.org/2021.findings-emnlp.35/)  | [PyTorch](https://github.com/RUCKBReasoning/P-INT)  |
| MetaP  | MetaP: Meta Pattern Learning for One-Shot Knowledge Graph Completion  | SIGIR 2021  | [PDF](https://dl.acm.org/doi/abs/10.1145/3404835.3463086)  | [PyTorch](https://github.com/jzystc/metap)  |
| MTransH  | Relational Learning with Gated and Attentive Neighbor Aggregator for Few-Shot Knowledge Graph Completion  | SIGIR 2021  | [PDF](https://arxiv.org/abs/2104.13095)  | [PyTorch](https://github.com/ngl567/GANA-FewShotKGC)  |
| KEFDA  | Knowledge-Enhanced Domain Adaptation in Few-Shot Relation Classification  | KDD 2021  | [PDF](https://dl.acm.org/doi/abs/10.1145/3447548.3467438)  | [PyTorch](https://github.com/imJiawen/KEFDA)  |
| IAN  | Multi-view Interaction Learning for Few-Shot Relation Classification  | CIKM 2021  | [PDF](https://dl.acm.org/doi/abs/10.1145/3459637.3482280)  | [N/A]  |
| HMNet  | HMNet: Hybrid Matching Network for Few-Shot Link Prediction  | DASFAA 2021  | [PDF](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_21)  | [N/A]  |
| APN-LW-JRL   | Adaptive Prototypical Networks with Label Words and Joint Representation Learning for Few-Shot Relation Classification  | TNNLS 2021  | [PDF](https://arxiv.org/abs/2101.03526)  | [N/A]  |
| GMUC  | Gaussian Metric Learning for Few-Shot Uncertain Knowledge Graph Completion  | DASFAA 2021  | [PDF](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_18)  | [PyTorch](https://github.com/zhangjiatao/GMUC)  |
| FAEA  | Function-words Enhanced Attention Networks for Few-Shot Inverse Relation Classification  | IJCAI 2022  | [PDF](https://arxiv.org/abs/2204.12111)  | [PyTorch](https://github.com/DOU123321/FAEA-FSRC)  |
| MULTIFORM  | MULTIFORM: Few-Shot Knowledge Graph Completion via Multi-modal Contexts  | ECML-PKDD 2022  | [PDF](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_354.pdf)  | [N/A]  |
| GAPNM  | Granularity-Aware Area Prototypical Network With Bimargin Loss for Few Shot Relation Classification  | TKDE 2022  | [PDF](https://ieeexplore.ieee.org/document/9699028)  | [N/A]  |
| Meta-iKG  | Subgraph-aware Few-Shot Inductive Link Prediction via Meta-Learning  | TKDE 2022  | [PDF](https://arxiv.org/abs/2108.00954)  | [N/A]  |
|   | Improving Few-Shot Relation Classification by Prototypical Representation Learning with Definition Text  | NAACL 2022  | [PDF](https://aclanthology.org/2022.findings-naacl.34/)  | [N/A]  |
| CIAN  | Learning Inter-Entity-Interaction for Few-Shot Knowledge Graph Completion  | EMNLP 2022  | [PDF](https://aclanthology.org/2022.emnlp-main.524/)  | [PyTorch](https://github.com/cjlyl/FKGC-CIAN)  |
| HiRe  | Hierarchical Relational Learning for Few-Shot Knowledge Graph Completion  | ICLR 2023  | [PDF](https://openreview.net/forum?id=zlwBI2gQL3K)  | [PyTorch](https://github.com/alexhw15/HiRe)  |
| NP-FKGC  | Normalizing Flow-based Neural Process for Few-Shot Knowledge Graph Completion  | SIGIR 2023  | [PDF](https://arxiv.org/abs/2304.08183)  | [PyTorch](https://github.com/RManLuo/NP-FKGC)  |



#### 2.2.2 Zero-Shot Relation Classification

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ZSGAN  | Generative Adversarial Zero-Shot Relational Learning for Knowledge Graphs  | AAAI 2020  | [PDF](https://arxiv.org/abs/2001.02332)  | [PyTorch](https://github.com/Panda0406/Zero-shot-knowledge-graph-relational-learning)  |
| ZSLRC  | Zero-shot Relation Classification from Side Information  | CIKM 2021  | [PDF](https://arxiv.org/abs/2011.07126)  | [PyTorch](https://github.com/gjiaying/ZSLRC)  |

#### 2.2.3 Few-Shot Reasoning on KGs

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Meta-KGR  | Adapting Meta Knowledge Graph Information for Multi-Hop Reasoning over Few-Shot Relations  | EMNLP-IJCNLP 2019  | [PDF](https://aclanthology.org/D19-1334/)  | [PyTorch](https://github.com/THU-KEG/MetaKGR)  |
| FIRE  | Few-Shot Multi-Hop Relation Reasoning over Knowledge Bases  | EMNLP 2020  | [PDF](https://aclanthology.org/2020.findings-emnlp.51/)  | [N/A]  |
| THML  | When Hardness Makes a Difference: Multi-Hop Knowledge Graph Reasoning over Few-Shot Relations  | CIKM 2021  | [PDF](https://dl.acm.org/doi/10.1145/3459637.3482402)  | [N/A]  |
| ADK-KG  | Adapting Distilled Knowledge for Few-shot Relation Reasoning over Knowledge Graphs  | SDM 2022  | [PDF](https://epubs.siam.org/doi/10.1137/1.9781611977172.75)  | [PyTorch](https://github.com/ADK-KG/ADK-KG)  |

### 2.3 Graph-Level Structure Imbalance

#### 2.3.1 Imbalanced Graph Sizes

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| SOLT-GNN  | On Size-Oriented Long-Tailed Graph Classification of Graph Neural Networks  | WWW 2022  | [PDF](https://zemin-liu.github.io/papers/SOLT-GNN-WWW-22.pdf)  | [PyTorch](https://github.com/shuaiOKshuai/SOLT-GNN)  |


#### 2.3.2 Imbalanced Topology Groups

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| TopoImb  | TopoImb: Toward Topology-level Imbalance in Learning from Graphs  | LoG 2022  | [PDF](https://arxiv.org/abs/2212.08689)  | [N/A]  |


## 3. Other Related Literature

### 3.1 Fairness Learning on Graphs

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |
| name  | title  | venue  | [PDF]()  | [PyTorch]()  |


# Acknowledgements
This page is contributed and maintained by Zemin Liu (zeminliu@nus.edu.sg), Yuan Li (xxxx), and Nan Chen (xxxx). If you have any suggestions or questions, please feel free to contact us.
