# Awesome Literature on [Imbalanced Learning on Graphs](xxx.xxx.xxx) (ILoGs)
This repository showcases a curated collection of research literature on imbalanced learning on graphs. We have categorized this literature according to the taxonomies of **Problems** and **Techniques** detailed in our survey paper, titled [A Survey of Imbalanced Learning on Graphs: Problems, Techniques, and Future Directions](xxx.xxx.xxx). In this repository, we primarily arrange the literature based on our **Problem** taxonomy for clarity. For a deeper understanding of this rapidly evolving and challenging field, we encourage readers to consult our survey.

For our taxonomy of **Problems**, we classify the literature based on **class imbalance** and **structure imbalance**, both stemming from imbalanced input. We further distill this into more specific categories: node-, edge-, and graph-level imbalance, offering a comprehensive understanding of graph imbalance.

For a more comprehensive overview of imbalanced learning on various data, please refer to Github Repository [Awesome-Imbalanced-Learning](https://github.com/yanliang3612/awesome-imbalanced-learning-on-geometric-and-graphs).


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
| Amplay  | Node re-ordering as a means of anomaly detection in time-evolving graphs  | ECML PKDD 2016  | [PDF](https://link.springer.com/chapter/10.1007/978-3-319-46227-1_11)  | [N/A]  |
|   | An embedding approach to anomaly detection  | ICDE 2016  | [PDF](https://ieeexplore.ieee.org/abstract/document/7498256/)  | [C++](https://github.com/hurenjun/EmbeddingAnomalyDetection)  |
| PFrauDetector  | PFrauDetector: A Parallelized Graph Mining Approach for Efficient Fraudulent Phone Call Detection  | ICPADS 2016  | [PDF](https://ieeexplore.ieee.org/abstract/document/7823855/)  | [N/A]  |
| HitFraud  | HitFraud: A Broad Learning Approach for Collective Fraud Detection in Heterogeneous Information Networks  | ICDM 2017  | [PDF](https://ieeexplore.ieee.org/abstract/document/8215553/)  | [N/A]  |
| FRAUDAR  | Graph-based fraud detection in the face of camouflage  | TKDD 2017  | [PDF](https://dl.acm.org/doi/abs/10.1145/3056563)  | [NumPy](https://bhooi.github.io/projects/fraudar/index.html)  |
| HiDDen  | Hidden: hierarchical dense subgraph detection with application to financial fraud detection  | SDM 2017  | [PDF](https://epubs.siam.org/doi/abs/10.1137/1.9781611974973.64)  | [MATLAB](https://github.com/sizhang92/HiDDen-SDM17)  |
| ALAD  | Accelerated Local Anomaly Detection via Resolving Attributed Networks  | IJCAI 2017  | [PDF](https://www.ijcai.org/Proceedings/2017/0325.pdf)  | [NumPy](https://github.com/ninghaohello/ALAD)  |
| GANG  | GANG: Detecting fraudulent users in online social networks via guilt-by-association on directed graphs  | ICDM 2017  | [PDF](https://home.engineering.iastate.edu/~neilgong/papers/GANG.pdf)  | [NumPy (Non-official)](https://github.com/safe-graph/UGFraud/blob/master/UGFraud/Detector/GANG.py)  |
| MTHL  | Anomaly detection in dynamic networks using multi-view time-series hypersphere learning  | CIKM 2017  | [PDF](https://xit22penny.github.io/files/pdf/research/2017-anomaly-detection.pdf)  | [NumPy](https://github.com/picsolab/Anomaly_Detection_MTHL)  |
|   | Spectrum-based deep neural networks for fraud detection  | CIKM 2017  | [PDF](https://arxiv.org/pdf/1706.00891.pdf)  | [N/A]  |
| Radar  | Radar: Residual Analysis for Anomaly Detection in Attributed Networks  | IJCAI 2017  | [PDF](https://www.ijcai.org/proceedings/2017/0299.pdf)  | [PyTorch](https://github.com/pygod-team/pygod)  |
| HoloScope  | HoloScope: Topology-and-Spike Aware Fraud Detection  | CIKM 2017  | [PDF](https://shenghua-liu.github.io/papers/cikm2017-holoscope.pdf)  | [NumPy](https://github.com/shenghua-liu/HoloScope)  |
| HoloScope  | A contrast metric for fraud detection in rich graphs  | TKDE 2018  | [PDF](https://ieeexplore.ieee.org/ielaam/69/8893432/8494803-aam.pdf)  | [NumPy](https://github.com/shenghua-liu/HoloScope)  |
| SEANO  | Semi-supervised embedding in attributed networks with outliers  | SDM 2018  | [PDF](https://arxiv.org/abs/1703.08100)  | [TensorFlow](http://jiongqianliang.com/SEANO/)  |
| Metagraph2vec  | Gotcha-sly malware! scorpion a metagraph2vec based malware detection system  | KDD 2018  | [PDF](https://dl.acm.org/doi/abs/10.1145/3219819.3219862)  | [N/A]  |
| PAICAN  | Bayesian robust attributed graph clustering: Joint learning of partial anomalies and group structure  | AAAI 2018  | [PDF](https://cdn.aaai.org/ojs/11642/11642-13-15170-1-2-20201228.pdf)  | [TensorFlow](https://github.com/abojchevski/paican)  |
| Netwalk  | NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks  | KDD 2018  | [PDF](https://www.researchgate.net/profile/Wei-Cheng-4/publication/329087157_A_Deep_Neural_Network_for_Unsupervised_Anomaly_Detection_and_Diagnosis_in_Multivariate_Time_Series_Data/links/5c7d7e04458515831f83ce81/A-Deep-Neural-Network-for-Unsupervised-Anomaly-Detection-and-Diagnosis-in-Multivariate-Time-Series-Data.pdf)  | [TensorFlow](https://github.com/chengw07/NetWalk)  |
| DeepSphere  | Deep into Hypersphere: Robust and Unsupervised Anomaly Discovery in Dynamic Networks  | IJCAI 2018  | [PDF](https://www.ijcai.org/Proceedings/2018/0378.pdf)  | [TensorFlow](https://github.com/picsolab/DeepSphere)  |
| NHAD  | NHAD: Neuro-Fuzzy Based Horizontal Anomaly Detection In Online Social Networks  | TKDE 2018  | [PDF](https://arxiv.org/abs/1804.06733)  | [N/A]  |
|   | Designing Size Consistent Statistics for Accurate Anomaly Detection in Dynamic Networks  | TKDD 2018  | [PDF](https://dl.acm.org/doi/abs/10.1145/3185059)  | [N/A]  |
| SPARC  | SPARC: Self-Paced Network Representation for Few-Shot Rare Category Characterization  | KDD 2018  | [PDF](https://dl.acm.org/doi/abs/10.1145/3219819.3219968)  | [Theano](https://sites.google.com/view/dawei-zhou/publications)  |
| ANOMALOUS  | ANOMALOUS: A Joint Modeling Approach for Anomaly Detection on Attributed Networks  | IJCAI 2018  | [PDF](https://www.ijcai.org/Proceedings/2018/0488.pdf)  | [MATLAB](https://github.com/zpeng27/ANOMALOUS)  |
| FFD  | Mining fraudsters and fraudulent strategies in large-scale mobile social networks  | TKDE 2019  | [PDF](https://ericdongyx.github.io/papers/TKDE19-yang-fraud-detection.pdf)  | [N/A]  |
| GraphUCB  | Interactive anomaly detection on attributed networks  | WSDM 2019  | [PDF](https://dl.acm.org/doi/abs/10.1145/3289600.3290964)  | [NumPy](https://github.com/kaize0409/GraphUCB_AnomalyDetection)  |
| Dominant  | Deep anomaly detection on attributed networks  | SDM 2019  | [PDF](https://www.researchgate.net/profile/Kaize-Ding/publication/332888297_Deep_Anomaly_Detection_on_Attributed_Networks/links/606f78364585150fe993abb6/Deep-Anomaly-Detection-on-Attributed-Networks.pdf)  | [PyTorch](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch)  |
| AnomRank  | Fast and accurate anomaly detection in dynamic graphs with a two-pronged approach  | KDD 2019  | [PDF](https://dl.acm.org/doi/abs/10.1145/3292500.3330946)  | [C++](https://github.com/minjiyoon/KDD19-AnomRank)  |
| ONE  | Outlier aware network embedding for attributed networks  | AAAI 2019  | [PDF](https://aaai.org/ojs/index.php/AAAI/article/download/3763/3641)  | [NetworkX](https://github.com/sambaranban/ONE)  |
| SpecAE  | SpecAE: Spectral AutoEncoder for Anomaly Detection in Attributed Networks  | CIKM 2019  | [PDF](https://dl.acm.org/doi/abs/10.1145/3357384.3358074)  | [N/A]  |
| QANet  | QANet: Tensor Decomposition Approach for Query-Based Anomaly Detection in Heterogeneous Information Networks  | TKDE 2019  | [PDF](https://ieeexplore.ieee.org/abstract/document/8488508/)  | [N/A]  |
| MADAN  | Multi-scale anomaly detection on attributed networks  | AAAI 2020  | [PDF](https://cdn.aaai.org/ojs/5409/5409-13-8634-1-10-20200511.pdf)  | [NetworkX](https://github.com/leoguti85/MADAN)  |
| ALARM  | A Deep Multi-View Framework for Anomaly Detection on Attributed Networks  | TKDE 2020  | [PDF](https://ieeexplore.ieee.org/abstract/document/9162509/)  | [PyTorch (Non-official)](https://github.com/Kaslanarian/SAGOD)  |
| MIDAS  | Midas: Microcluster-based detector of anomalies in edge streams  | AAAI 2020  | [PDF](https://cdn.aaai.org/ojs/5724/5724-13-8949-1-10-20200513.pdf)  | [C++](https://github.com/Stream-AD/MIDAS)  |
| GraphRfi  | Gcn-based user representation learning for unifying robust recommendation and fraudster detection  | SIGIR 2020  | [PDF](https://www.researchgate.net/profile/Tong-Chen-64/publication/341539062_GCN-Based_User_Representation_Learning_for_Unifying_Robust_Recommendation_and_Fraudster_Detection/links/6100eb6c169a1a0103bf9a9b/GCN-Based-User-Representation-Learning-for-Unifying-Robust-Recommendation-and-Fraudster-Detection.pdf)  | [PyTorch](https://github.com/zsjdddhr/GraphRfi)  |
| BEA  | Anomaly Detection on Dynamic Bipartite Graph with Burstiness  | ICDM 2020  | [PDF](https://www.researchgate.net/profile/Aixin-Sun/publication/349201957_Anomaly_Detection_on_Dynamic_Bipartite_Graph_with_Burstiness/links/6143f0c0a609b152aa157610/Anomaly-Detection-on-Dynamic-Bipartite-Graph-with-Burstiness.pdf)  | [N/A]  |
| MixedAD  | Mixedad: a scalable algorithm for detecting mixed anomalies in attributed graphs  | AAAI 2020  | [PDF](https://cdn.aaai.org/ojs/5482/5482-13-8707-1-10-20200511.pdf)  | [N/A]  |
| STAGN  | Graph neural network for fraud detection via spatial-temporal attention  | TKDE 2020  | [PDF](https://ieeexplore.ieee.org/abstract/document/9204584/)  | [PyTorch](https://github.com/finint/antifraud)  |
| GAL  | Error-bounded graph anomaly loss for GNNs  | CIKM 2020  | [PDF](https://www.researchgate.net/profile/Tong-Zhao-20/publication/346275948_Error-Bounded_Graph_Anomaly_Loss_for_GNNs/links/5fc92515299bf188d4f13924/Error-Bounded-Graph-Anomaly-Loss-for-GNNs.pdf)  | [PyTorch](https://github.com/zhao-tong/Graph-Anomaly-Loss)  |
| GAAN  | Generative Adversarial Attributed Network Anomaly Detection  | CIKM 2020  | [PDF](https://static.aminer.cn/storage/pdf/acm/20/cikm/10.1145/3340531.3412070.pdf)  | [PyTorch](https://github.com/Kaslanarian/SAGOD)  |
| DMGD  | Integrating Network Embedding and Community Outlier Detection via Multiclass Graph Description  | ECAI 2020  | [PDF](https://ecai2020.eu/papers/1161_paper.pdf)  | [TensorFlow](https://github.com/vasco95/DMGD)  |
| CARE-GNN  | Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters  | CIKM 2020  | [PDF](https://penghao-bdsc.github.io/papers/cikm20.pdf)  | [PyTorch](https://github.com/YingtongDou/CARE-GNN)  |
| GraphConsis  | Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection  | SIGIR 2020  | [PDF](https://par.nsf.gov/servlets/purl/10167818)  | [TensorFlow](https://github.com/safe-graph/DGFraud)  |
| COSIN  | Fraud Detection in Dynamic Interaction Network  | TKDE 2020  | [PDF](https://drive.google.com/file/d/1F6mOqzpkDp_2zhi7vvoZCuWByiFfKs5j/view)  | [N/A]  |
| HMGNN  | Heterogeneous Mini-Graph Neural Network and Its Application to Fraud Invitation Detection  | ICDM 2020  | [PDF](https://ieeexplore.ieee.org/abstract/document/9338325/)  | [TensorFlow](https://github.com/iqiyi/HMGNN)  |
| C-FATH  | Modeling Heterogeneous Graph Network on Fraud Detection: A Community-based Framework with Attention Mechanism  | CIKM 2021  | [PDF](https://dl.acm.org/doi/abs/10.1145/3459637.3482277)  | [N/A]  |
| A  | Graph Regularized Autoencoder and its Application in Unsupervised Anomaly Detection  | TPAMI 2021  | [PDF](https://ieeexplore.ieee.org/abstract/document/9380495/)  | [N/A]  |
| DCI  | Decoupling Representation Learning and Classification for GNN-based Anomaly Detection  | SIGIR 2021  | [PDF](https://xiaojingzi.github.io/publications/SIGIR21-Wang-et-al-decoupled-GNN.pdf)  | [PyTorch](https://github.com/wyl7/DCI-pytorch)  |
| IHGAT  | Intention-aware Heterogeneous Graph Attention Networks for Fraud Transactions Detection  | KDD 2021  | [PDF](https://dl.acm.org/doi/abs/10.1145/3447548.3467142)  | [N/A]  |
| AAGNN  | Subtractive Aggregation for Attributed Network Anomaly Detection  | CIKM 2021  | [PDF](https://dl.acm.org/doi/abs/10.1145/3459637.3482195)  | [PyTorch](https://github.com/betterzhou/AAGNN)  |
| ANEMONE  | ANEMONE: Graph Anomaly Detection with Multi-Scale Contrastive Learning  | CIKM 2021  | [PDF](https://shiruipan.github.io/publication/cikm-21-jin/cikm-21-jin.pdf)  | [PyTorch](https://github.com/GRAND-Lab/ANEMONE)  |
| TDA  | Topological anomaly detection in dynamic multilayer blockchain networks  | ECML PKDD 2021  | [PDF](https://link.springer.com/chapter/10.1007/978-3-030-86486-6_48)  | [R](https://github.com/tdagraphs/tdagraphs)  |
| SkewA  | Graph Fraud Detection Based on Accessibility Score Distributions  | ECML PKDD 2021  | [PDF](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_851.pdf)  | [C++](https://github.com/minjiyoon/PKDD21-SkewA)  |
| BiDyn  | Bipartite Dynamic Representations for Abuse Detection  | KDD 2021  | [PDF](https://dl.acm.org/doi/abs/10.1145/3447548.3467141)  | [PyTorch](https://github.com/qema/bidyn)  |
| GRC  | Towards Consumer Loan Fraud Detection: Graph Neural Networks with Role-Constrained Conditional Random Field  | AAAI 2021  | [PDF](https://aaai.org/AAAI21Papers/AAAI-6859.XuB.pdf)  | [N/A]  |
|   | Signature-Based Anomaly Detection in Networks  | SDM 2021  | [PDF](https://epubs.siam.org/doi/abs/10.1137/1.9781611976700.13)  | [N/A]  |
| CO-GCN  | Graph Neural Network to Dilute Outliers for Refactoring Monolith Application  | AAAI 2021  | [PDF](https://cdn.aaai.org/ojs/16079/16079-13-19573-1-2-20210518.pdf)  | [PyTorch](https://github.com/utkd/cogcn)  |
| PAMFUL  | A Synergistic Approach for Graph Anomaly Detection With Pattern Mining and Feature Learning  | TNNLS 2021  | [PDF](https://ieeexplore.ieee.org/abstract/document/9525041/)  | [PyTorch](https://github.com/zhao-tong/Graph-Anomaly-Loss)  |
| COMMANDER  | Cross-Domain Graph Anomaly Detection  | TNNLS 2021  | [PDF](https://par.nsf.gov/servlets/purl/10352642)  | [N/A]  |
| GDN  | Graph Neural Network-Based Anomaly Detection in Multivariate Time Series  | AAAI 2021  | [PDF](https://cdn.aaai.org/ojs/16523/16523-13-20017-1-2-20210518.pdf)  | [PyTorch](https://github.com/d-ailin/GDN)  |
| PC-GNN  | Pick and choose: a GNN-based imbalanced learning approach for fraud detection  | WWW 2021  | [PDF](https://ponderly.github.io/pub/PCGNN_WWW2021.pdf)  | [PyTorch](https://github.com/PonderLY/PC-GNN)  |
| GDN  | Few-shot network anomaly detection via cross-network meta-learning  | WWW 2021  | [PDF](https://dl.acm.org/doi/abs/10.1145/3442381.3449922)  | [PyTorch](https://github.com/kaize0409/Meta-GDN_AnomalyDetection)  |
| CoLA  | Anomaly detection on attributed networks via contrastive self-supervised learning  | TNNLS 2021  | [PDF](https://ieeexplore.ieee.org/abstract/document/9395172/)  | [PyTorch](https://github.com/grand-lab/cola)  |
| AEGIS  | Inductive Anomaly Detection on Attributed Networks  | IJCAI 2021  | [PDF](https://dl.acm.org/doi/abs/10.5555/3491440.3491619)  | [N/A]  |
| Turbo  | Turbo: Fraud Detection in Deposit-free Leasing Service via Real-Time Behavior Network Mining  | ICDE 2021  | [PDF](https://ieeexplore.ieee.org/abstract/document/9458859/)  | [N/A]  |
| MetaHG  | Distilling Meta Knowledge on Heterogeneous Graph for Illicit Drug Trafficker Detection on Social Media  | NeurIPS 2021  | [PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/e234e195f3789f05483378c397db1cb5-Supplemental.pdf)  | [PyTorch](https://github.com/graphprojects/MetaHG)  |
| EnsemFDet  | EnsemFDet: An Ensemble Approach to Fraud Detection based on Bipartite Graph  | ICDE 2021  | [PDF](https://www.researchgate.net/profile/Hao-Zhu-35/publication/338158171_EnsemFDet_An_Ensemble_Approach_to_Fraud_Detection_based_on_Bipartite_Graph/links/604619ff4585154e8c864b34/EnsemFDet-An-Ensemble-Approach-to-Fraud-Detection-based-on-Bipartite-Graph.pdf)  | [N/A]  |
| FRAUDRE  | FRAUDRE: Fraud Detection Dual-Resistant to Graph Inconsistency and Imbalance  | ICDM 2021  | [PDF](https://www.researchgate.net/profile/Chuan-Zhou-3/publication/357512222_FRAUDRE_Fraud_Detection_Dual-Resistant_to_Graph_Inconsistency_and_Imbalance/links/61d18807b8305f7c4b19bd14/FRAUDRE-Fraud-Detection-Dual-Resistant-to-Graph-Inconsistency-and-Imbalance.pdf)  | [PyTorch](https://github.com/FraudDetection/FRAUDRE)  |
| ELAND  | Action Sequence Augmentation for Early Graph-based Anomaly Detection  | CIKM 2021  | [PDF](https://arxiv.org/abs/2010.10016)  | [PyTorch](https://github.com/dm2-nd/eland)  |
| DeFraudNet  | DeFraudNet: An End-to-End Weak Supervision Framework to Detect Fraud in Online Food Delivery  | ECML PKDD 2021  | [PDF](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_10-1.pdf)  | [N/A]  |
| DIGNN  | The Devil is in the Conflict: Disentangled Information Graph Neural Networks for Fraud Detection  | ICDM 2022  | [PDF](https://arxiv.org/abs/2210.12384)  | [N/A]  |
| STGAN  | Graph Convolutional Adversarial Networks for Spatiotemporal Anomaly Detection  | TNNLS 2022  | [PDF](https://ieeexplore.ieee.org/abstract/document/9669110/)  | [PyTorch](https://github.com/dleyan/STGAN)  |
| DAGAD  | DAGAD: Data Augmentation for Graph Anomaly Detection  | ICDM 2022  | [PDF](https://ieeexplore.ieee.org/abstract/document/10027747/)  | [PyTorch](https://github.com/fanzhenliu/dagad)  |
| NGS  | Explainable Graph-based Fraud Detection via Neural Meta-graph Search  | CIKM 2022  | [PDF](https://ponderly.github.io/pub/NGS_CIKM2022.pdf)  | [N/A]  |
| BRIGHT  | BRIGHT-Graph Neural Networks in Real-time Fraud Detection  | CIKM 2022  | [PDF](https://dl.acm.org/doi/abs/10.1145/3511808.3557136)  | [N/A]  |
| Hetero-SCAN  | Meta-Path-based Fake News Detection Leveraging Multi-level Social Context Information  | CIKM 2022  | [PDF](https://arxiv.org/abs/2109.08022)  | [N/A]  |
| DynAnom  | Subset Node Anomaly Tracking over Large Dynamic Graphs  | KDD 2022  | [PDF](https://openreview.net/pdf?id=Xx7eJDXG3Xl)  | [NetworkX](https://github.com/zjlxgxz/dynanom)  |
| H2-FDetector  | H2-FDetector: A GNN-based Fraud Detector with Homophilic and Heterophilic Connections  | WWW 2022  | [PDF](https://scholar.archive.org/work/fomltdkxnrblndckrapxjyusri/access/wayback/https://dl.acm.org/doi/pdf/10.1145/3485447.3512195)  | [PyTorch](https://github.com/shifengzhao/H2-FDetector)  |
| AO-GNN  | AUC-oriented Graph Neural Network for Fraud Detection  | WWW 2022  | [PDF](https://ponderly.github.io/pub/AOGNN_WWW2022.pdf)  | [N/A]  |
| ComGA  | ComGA: Community-Aware Attributed Graph Anomaly Detection  | WSDM 2022  | [PDF](https://dl.acm.org/doi/abs/10.1145/3488560.3498389)  | [TensorFlow](https://github.com/XuexiongLuoMQ/ComGA)  |
| Sub-CR  | Reconstruction Enhanced Multi-View Contrastive Learning for Anomaly Detection on Attributed Networks  | IJCAI 2022  | [PDF](https://www.ijcai.org/proceedings/2022/0330.pdf)  | [PyTorch](https://github.com/Zjer12/Sub)  |
| LUNAR  | LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks  | AAAI 2022  | [PDF](https://cdn.aaai.org/ojs/20629/20629-13-24642-1-2-20220628.pdf)  | [PyTorch](https://github.com/agoodge/lunar)  |
| DVGCRN  | Deep Variational Graph Convolutional Recurrent Network for Multivariate Time Series Anomaly Detection  | ICML 2022  | [PDF](https://scholar.archive.org/work/gsvjxkzpqrch7eru5jcwa7mkcq/access/wayback/https://proceedings.mlr.press/v162/chen22x/chen22x.pdf)  | [PyTorch](https://github.com/BoChenGroup/DVGCRN)  |
| MAD-SGCN  | MAD-SGCN: Multivariate Anomaly Detection with Self-learning Graph Convolutional Networks  | ICDE 2022  | [PDF](https://ieeexplore.ieee.org/abstract/document/9835470/)  | [N/A]  |
| MHGL  | Unseen Anomaly Detection on Networks via Multi-Hypersphere Learning  | SDM 2022  | [PDF](https://www4.comp.polyu.edu.hk/~xiaohuang/docs/Shuang_SDM22.pdf)  | [N/A]  |
| HCM  | Hop-Count Based Self-Supervised Anomaly Detection on Attributed Networks  | ECML PKDD 2022  | [PDF](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_927.pdf)  | [PyTorch](https://github.com/TienjinHuang/GraphAnomalyDetection)  |
| BWGNN  | Rethinking Graph Neural Networks for Anomaly Detection  | ICML 2022  | [PDF](https://www.researchgate.net/profile/Jia-Li-127/publication/360994234_Rethinking_Graph_Neural_Networks_for_Anomaly_Detection/links/6299d59b6886635d5cbb9bb1/Rethinking-Graph-Neural-Networks-for-Anomaly-Detection.pdf)  | [PyTorch](https://github.com/squareroot3/rethinking-anomaly-detection)  |
| BLS  | Bi-Level Selection via Meta Gradient for Graph-Based Fraud Detection  | DASFAA 2022  | [PDF](https://ponderly.github.io/pub/BLS_DASFAA2022.pdf)  | [N/A]  |
| GraphAD  | GraphAD: A Graph Neural Network for Entity-Wise Multivariate Time-Series Anomaly Detection  | SIGIR 2022  | [PDF](https://arxiv.org/abs/2205.11139)  | [N/A]  |
| GAGA  | Label Information Enhanced Fraud Detection against Low Homophily in Graphs  | WWW 2023  | [PDF](https://arxiv.org/abs/2302.10407)  | [PyTorch](https://github.com/orion-wyc/gaga)  |
| GHRN  | Addressing Heterophily in Graph Anomaly Detection: A Perspective of Graph Spectrum  | WWW 2023  | [PDF](https://hexiangnan.github.io/papers/www23-graphAD.pdf)  | [PyTorch](https://github.com/blacksingular/GHRN)  |
| GDN  | Alleviating Structural Distribution Shift in Graph Anomaly Detection  | WSDM 2023  | [PDF](https://hexiangnan.github.io/papers/wsdm23-GDN.pdf)  | [PyTorch](https://github.com/blacksingular/wsdm_GDN)  |
| CODEtect  | Detecting Anomalous Graphs in Labeled Multi-Graph Databases  | TKDD 2023  | [PDF](https://scholar.archive.org/work/lx4zgpvoezgbbkemvkbywfbhmm/access/wayback/https://dl.acm.org/doi/pdf/10.1145/3533770)  | [N/A]  |
| GCAD  | Subgraph Centralization: A Necessary Step for Graph Anomaly Detection  | SDM 2023  | [PDF](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch79)  | [NumPy](https://github.com/IsolationKernel/Codes)  |
| SAD  | SAD: Semi-Supervised Anomaly Detection on Dynamic Graphs  | IJCAI 2023  | [PDF](https://arxiv.org/abs/2305.13573)  | [NumPy](https://github.com/d10andy/sad)  |
| VGOD  | Unsupervised Graph Outlier Detection: Problem Revisit, New Insight, and Superior Method  | ICDE 2023  | [PDF](https://openreview.net/pdf?id=Kh5gknUMBk)  | [PyTorch](https://github.com/goldennormal/vgod-github)  |


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
| AddGraph  | AddGraph: Anomaly Detection in Dynamic Graph Using Attention-based Temporal GCN | IJCAI 2019 | [PDF](https://www.ijcai.org/Proceedings/2019/0614.pdf)  | [PyTorch](https://github.com/Ljiajie/Addgraph)  |
| NEDM  | A Nodes' Evolution Diversity Inspired Method to Detect Anomalies in Dynamic Social Networks | TKDE 2019 | [PDF](https://ieeexplore.ieee.org/document/8695818)  | [N/A]  |
|  | Anomaly detection in the dynamics of web and social networks using associative memory  | WWW 2019 | [PDF](https://arxiv.org/pdf/1901.09688.pdf)  | [N/A]  |
| AANE  | AANE: Anomaly Aware Network Embedding for Anomalous Link Detection  | ICDM 2020 | [PDF](https://ieeexplore.ieee.org/document/9338406)  | [N/A]  |
| StrGNN  | Structural Temporal Graph Neural Networks for Anomaly Detection in Dynamic Graphs  | CIKM 2021  | [PDF](https://arxiv.org/pdf/2005.07427.pdf)  | [PyTorch](https://github.com/LeiCaiwsu/StrGNN)  |
| F-FADE  | F-FADE: Frequency Factorization for Anomaly Detection in Edge Streams  | WSDM 2021  | [PDF](https://arxiv.org/pdf/2011.04723.pdf)  | [PyTorch](https://github.com/snap-stanford/F-FADE)  |
| LIFE  | Live-Streaming Fraud Detection: A Heterogeneous Graph Neural Network Approach  | KDD 2021  | [PDF](https://dl.acm.org/doi/abs/10.1145/3447548.3467065)  | [N/A]  |
| GLAD  | Deep Graph Learning for Anomalous Citation Detection  | TNNLS 2022 | [PDF](https://arxiv.org/abs/2202.11360)  | [N/A] |

### 1.3 Graph-Level Class Imbalance

#### 1.3.1 Imbalanced Graph Classification

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| $\text{G}^2\text{GNN}$  | Imbalanced Graph Classification via Graph-of-Graph Neural Networks | CIKM 2022  | [PDF](https://arxiv.org/abs/2112.00238)  | [PyTorch](https://github.com/YuWVandy/G2GNN)  |


#### 1.3.2 Graph-Level Anomaly Detection

| Name  | Title | Venue | Paper | Code |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| STREAMSPOT  | Fast Memory-efficient Anomaly Detection in Streaming Heterogeneous Graphs  | KDD 2016 | [PDF](https://arxiv.org/pdf/1602.04844.pdf)  | [Python](https://git.ece.iastate.edu/yizenov1/stream-graph-anomaly-detection)  |
| Graph-TPP  | Query-Driven Discovery of Anomalous Subgraphs in Attributed Graphs  | IJCAI 2017  | [PDF](https://www.ijcai.org/proceedings/2017/0433.pdf)  | [N/A]  |
| AMEN  | Discovering Communities and Anomalies in Attributed Graphs: Interactive Visual Exploration and Summarization  | TKDD 2018  | [PDF](https://dl.acm.org/doi/10.1145/3139241)  | [N/A] |
|  | Concept Drift and Anomaly Detection in Graph Streams | TNNLS 2018  | [PDF](https://arxiv.org/abs/1706.06941)  | [N/A]  |
| Query-map  | Uncovering specific-shape graph anomalies in attributed graphs  | AAAI 2019  | [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/4483)  | [N/A]  |
| ASD-FT  | Anomaly Subgraph Detection with Feature Transfer  | CIKM 2020  | [PDF](https://dl.acm.org/doi/abs/10.1145/3340531.3411968)  | [N/A]  |
| MTAD-GAT | Multivariate Time-series Anomaly Detection via Graph Attention Network  | ICDM 2020  | [PDF](https://arxiv.org/abs/2009.02040)  | [TensorFlow](https://github.com/mangushev/mtad-gat)  |
| GraphAnoGAN  | GraphAnoGAN: Detecting Anomalous Snapshots from Attributed Graphs  | PKDD 2021 | [PDF](https://arxiv.org/abs/2106.15504)  | [TensorFlow](https://github.com/LCS2-IIITD/GraphAnoGAN-ECMLPKDD21/tree/main)  |
| GLocalKD  | Deep Graph-level Anomaly Detection by Glocal Knowledge Distillation  | WSDM 2022  | [PDF](https://arxiv.org/abs/2112.10063)  | [PyTorch](https://github.com/RongrongMa/GLocalKD)  |
| OCGTL  | Raising the Bar in Graph-level Anomaly Detection  | IJCAI 2022  | [PDF](https://arxiv.org/abs/2205.13845)  | [PyTorch](https://github.com/boschresearch/GraphLevel-AnomalyDetection)  |
| AS-GA  | Unsupervised Deep Subgraph Anomaly Detection  | CIKM 2022  | [PDF](https://ieeexplore.ieee.org/document/10027633)  | [PyTorch](https://github.com/rollingstonezz/subgraph_anomaly_detection_icdm22)  |
| AntiBenford  | AntiBenford Subgraphs: Unsupervised Anomaly Detection in Financial Networks  | KDD 2022  | [PDF](https://arxiv.org/abs/2205.13426)  | [Python](https://github.com/tsourakakis-lab/antibenford-subgraphs)  |
| CNSS  | Calibrated Nonparametric Scan Statistics for Anomalous Pattern Detection in Graphs  | AAAI 2022  | [PDF](https://arxiv.org/abs/2206.12786)  | [N/A]  |
| ACGPMiner  | Efficient Anomaly Detection in Property Graphs  | DASFAA 2023  | [PDF](https://link.springer.com/chapter/10.1007/978-3-031-30675-4_9)  | [N/A]  |

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
| CFC  | Compositional fairness constraints for graph embeddings  |  ICML 2019  | [PDF](https://arxiv.org/abs/1905.10674)  | [PyTorch](https://github.com/joeybose/Flexible-Fairness-Constraints)  |
|   | Exploring algorithmic fairness in robust graph covering problems  | NIPS 2019  | [PDF](https://arxiv.org/abs/2006.06865)  | [PyTorch](https://github.com/Aida-Rahmattalabi/FairGraphCovering)  |
| Fairwalk  | Fairwalk: Towards fair graph embedding  | IJCAI 2019  | [PDF](https://www.ijcai.org/proceedings/2019/456)  | [Python](https://github.com/EnderGed/Fairwalk)  |
|   | Spectral relaxations and fair densest subgraphs  | CIKM 2020  | [PDF](https://dl.acm.org/doi/10.1145/3340531.3412036)  | [N/A]  |
|   | Fairness-aware explainable recommendation over knowledge graphs  | SIGIR 2020  | [PDF](https://dl.acm.org/doi/abs/10.1145/3397271.3401051)  | [N/A](https://github.com/zuohuif/FairKG4Rec)  |
| MaxFair  | On the information unfairness of social networks  | SDM 2020 | [PDF](https://epubs.siam.org/doi/abs/10.1137/1.9781611976236.69)  | [N/A]  |
| FairGNN  | Say no to the discrimination: Learning fair graph neural networks with limited sensitive attribute information  | WSDM 2021  | [PDF](https://arxiv.org/abs/2009.01454)  | [PyTorch](https://github.com/EnyanDai/FairGNN)  |
| InFoRM  | Inform: Individual fairness on graph mining  | KDD 2020  | [PDF](https://dl.acm.org/doi/abs/10.1145/3394486.3403080)  | [Python](https://github.com/jiank2/inform)  |
| DeBayes  | Debayes: a bayesian method for debiasing network embeddings  | ICML 2020  | [PDF](https://arxiv.org/abs/2002.11442)  | [Python](https://github.com/aida-ugent/DeBayes)  |
| MLSD  | Fairness in network representation by latent structural heterogeneity in observational data  | AAAI 2020  | [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/5792)  | [N/A] |
| REDRESS  | Individual fairness for graph neural networks: A ranking based approach  | KDD 2021  | [PDF](https://dl.acm.org/doi/10.1145/3447548.3467266)  | [TensorFlow](https://github.com/yushundong/REDRESS)  |
| FairGAE  | Fair graph auto-encoder for unbiased graph representations with wasserstein distance  | ICDM 2021  | [PDF](https://ieeexplore.ieee.org/document/9679109)  | [N/A]  |
| MCCNIFTY  | A multi-view confidence-calibrated framework for fair and stable graph representation learning  | ICDM 2021  | [PDF](https://ieeexplore.ieee.org/document/9679093)  | [N/A]  |
|  | Certification and trade-off of multiple fairness criteria in graph-based spam detection  | CIKM 2021  | [PDF](https://dl.acm.org/doi/abs/10.1145/3459637.3482325)  | [N/A]  |
| Fairness-Aware PageRank  | Fairness-aware pagerank  | WWW 2021 | [PDF](https://arxiv.org/abs/2005.14431)  | [Python & C++](https://github.com/SotirisTsioutsiouliklis/FairLaR)  |
| FairAdj  | On dyadic fairness: Exploring and mitigating bias in graph connections  | ICLR 2021  | [PDF](https://openreview.net/forum?id=xgGS6PmzNq6)  | [PyTorch](https://github.com/brandeis-machine-learning/FairAdj)  |
|  | Subgroup generalization and fair- ness of graph neural networks | NIPS 2021  | [PDF](https://arxiv.org/abs/2106.15535)  | [PyTorch](https://github.com/TheaperDeng/GNN-Generalization-Fairness)  |
| MMSS  | Socially fair mitigation of misinformation on social networks via constraint stochastic optimization  | AAAI 2022  | [PDF](https://arxiv.org/abs/2203.12537)  | [Python](https://github.com/Ahmed-Abouzeid/MMSS)  |
| CrossWalk  | Crosswalk: Fairness-enhanced node representation learning  | AAAI 2022  | [PDF](https://arxiv.org/abs/2105.02725)  | [Scikit-Learn](https://github.com/ahmadkhajehnejad/CrossWalk)  |
| FairDrop  | Fairdrop: Biased edge dropout for enhancing fairness in graph representation learning  | IEEE Trans. Artif. Intell. 2022  | [PDF](https://arxiv.org/abs/2104.14210)  | [PyTorch](https://github.com/ispamm/FairDrop)  |
| FairVGNN  | Improving fairness in graph neural networks via mitigating sensitive attribute leakage  | KDD 2022  | [PDF](https://arxiv.org/abs/2206.03426)  | [PyTorch](https://github.com/YuWVandy/FairVGNN)  |
| GUIDE  | Guide: Group equality informed individual fairness in graph neural networks  | KDD 2022  | [PDF](https://dl.acm.org/doi/abs/10.1145/3534678.3539346)  | [PyTorch](https://github.com/weihaosong/GUIDE)  |
| REFEREE  | On structural explanation of bias in graph neural networks  | KDD 2022  | [PDF](https://arxiv.org/abs/2206.12104)  | [PyTorch](https://github.com/yushundong/REFEREE)  |
| UD-GNN  | UD-GNN: uncertainty-aware debiased training on semi-homophilous graphs  | KDD 2022  | [PDF](https://dl.acm.org/doi/10.1145/3534678.3539483)  | [N/A](https://github.com/PonderLY/UD-GNN)  |
| GEAR  | Learning fair node representations with graph counterfactual fairness  | WSDM 2022 | [PDF](https://arxiv.org/abs/2201.03662)  | [PyTorch](https://github.com/jma712/gear)  |
| EDITS  | Edits: Modeling and mitigating data bias for graph neural networks  | WWW 2022  | [PDF](https://arxiv.org/abs/2108.05233)  | [PyTorch](https://github.com/yushundong/EDITS)  |
| UGE  | Unbiased graph embedding with biased graph observations  | WWW 2022  | [PDF](https://arxiv.org/abs/2110.13957)  | [PyTorch](https://github.com/MyTHWN/UGE-Unbiased-Graph-Embedding)  |
|   | Adversarial inter-group link injection degrades the fairness of graph neural networks  | ICDM 2022  | [PDF](https://arxiv.org/abs/2209.05957)  | [PyTorch](https://github.com/mengcao327/attack-gnn-fairness)  |
| BA-GNN  | Ba-gnn:On learning bias-aware graph neural network  | ICDE 2022  | [PDF](https://ieeexplore.ieee.org/document/9835653)  | [N/A]  |
| FairAC  | Fair attribute completion on graph with missing attributes  | ICLR 2023  | [PDF](https://arxiv.org/abs/2302.12977)  | [PyTorch](https://github.com/donglgcn/FairAC)  |
| Graphair  | Learning fair graph representations via automated data augmentations  | ICLR 2023  | [PDF](https://openreview.net/forum?id=1_OGWcP1s9w)  | [PyTorch](https://github.com/divelab/DIG)  |
| FAIRTCIM  | On the fairness of time-critical influence maximization in social networks  | TKDE 2023  | [PDF](https://arxiv.org/abs/1905.06618)  | [N/A] |
| CGF  | Path-specific causal fair prediction via auxiliary graph structure learning  | WWW 2023  | [PDF](https://dl.acm.org/doi/abs/10.1145/3543507.3583280)  | [N/A]  |
| F-SEGA  | Fairness-aware clique-preserving spectral clustering of temporal graphs  | WWW 2023  | [PDF](https://dl.acm.org/doi/10.1145/3543507.3583423)  | [N/A](https://github.com/DongqiFu/F-SEGA)  |
| G-Fame  | Fair graph representation learning via diverse mixture-of-experts  | WWW 2023  | [PDF](https://dl.acm.org/doi/abs/10.1145/3543507.3583207)  | [N/A]  |
| RELIANT  | RELIANT: Fair Knowledge Distillation for Graph Neural Networks  | ICDM 2023  | [PDF](https://arxiv.org/abs/2301.01150)  | [PyTorch](https://github.com/yushundong/reliant)  |

# Acknowledgements
This page is contributed and maintained by Zemin Liu (zeminliu@nus.edu.sg), Yuan Li (li.yuan@nus.edu.sg), and Nan Chen (nanchen@nus.edu.sg). If you have any suggestions or questions, please feel free to contact us.
