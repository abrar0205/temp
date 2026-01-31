# Probabilistic Blood Type Inference Using Bayesian Networks

## Abstract

This report presents a Bayesian network approach for inferring blood type distributions in family trees with incomplete information. The model represents each person's genetic alleles and observable blood type as separate random variables connected by causal edges that reflect biological inheritance. Three evidence types are handled: direct blood tests, mixed-sample tests, and paired tests with potential labeling errors. Evaluation on test cases of varying complexity confirms correct probability inference. The key insight is that causal modeling (edges from genotype to phenotype) produces cleaner specifications than diagnostic modeling, even when the inference goal is to determine causes from observed effects.

## 1 Introduction

Blood type inference from partial family data is a classic problem in probabilistic reasoning. The ABO system follows clear genetic rules—three alleles (A, B, O) combine to produce four blood types—yet practical inference must handle uncertainty: unknown genotypes, missing relatives, and imperfect tests.

This report investigates whether Bayesian networks can effectively model this problem. The scenario involves families in Wumponia, a fictional country with two regions having different allele frequencies, and three types of laboratory tests with varying reliability.

The main contributions are: (1) a network architecture separating genetic state from observable phenotype, (2) methods for incorporating different evidence types, and (3) insights about causal versus diagnostic modeling. Section 2 provides background, Section 3 describes the model, Section 4 presents results, and Section 5 discusses findings.

## 2 Background

### 2.1 ABO Genetics

Each person has two ABO alleles inherited from their parents. Alleles A and B are codominant; O is recessive. The genotype-to-phenotype mapping is:

| Genotype | Blood Type |
|----------|------------|
| AA, AO   | A          |
| BB, BO   | B          |
| AB       | AB         |
| OO       | O          |

This mapping is deterministic but not invertible—blood type A could be AA or AO. Each parent passes one randomly-selected allele to each child with equal probability.

### 2.2 Population Priors

Wumponia has two regions with different allele frequencies:

| Region | P(A) | P(B) | P(O) |
|--------|------|------|------|
| North  | 0.50 | 0.25 | 0.25 |
| South  | 0.15 | 0.55 | 0.30 |

These serve as priors for founder individuals (those without known parents).

### 2.3 Bayesian Networks

A Bayesian network represents a joint distribution using a directed acyclic graph where nodes are random variables and edges encode dependencies. Each node has a conditional probability table (CPT) specifying its distribution given parent values. The key property: edges reflect causal direction, but inference can compute any conditional probability regardless of edge direction.

## 3 Model Architecture

### 3.1 Person Representation

Each individual has three nodes:
- **Allele1, Allele2**: The two inherited alleles (values: A, B, O)
- **BloodType**: Observable phenotype (values: A, B, AB, O)

BloodType has edges from both allele nodes with a deterministic CPT encoding the genotype-phenotype mapping.

### 3.2 Inheritance Structure

For individuals with known parents, I introduce intermediate "contribution" nodes. The father's contribution node depends on his two alleles; the child's Allele1 depends on this contribution. Similarly for mother and Allele2.

```
Father_Allele1 ──┬── Father_Contrib ──► Child_Allele1 ──┬── Child_BloodType
Father_Allele2 ──┘                                      │
                                       Child_Allele2 ──┘
Mother_Allele1 ──┬── Mother_Contrib ──► Child_Allele2
Mother_Allele2 ──┘
```

The contribution CPT encodes random selection: if parental alleles differ, each is passed with probability 0.5.

### 3.3 Evidence Types

**Standard tests** set the BloodType node to its observed value directly.

**Mixed tests** combine blood from two people. A new node represents the mixture result, with edges from both BloodType nodes. The CPT is deterministic: the mixture shows A if either person has A antigen, B if either has B.

**Paired tests** report both individuals' types, but labels may swap with 20% probability. A joint node with 16 states (all result pairs) captures the correlation—both labels are correct (80%) or both swapped (20%).

### 3.4 Construction Algorithm

The network is built by: (1) creating nodes for all individuals, (2) adding inheritance edges in topological order, (3) adding test nodes, (4) defining CPTs using inheritance rules and population priors, (5) setting evidence, and (6) running variable elimination for queries.

## 4 Evaluation

### 4.1 Test Categories

Problems span four categories: A (minimal families, standard tests), B (larger families, both regions), C (mixed tests), D (paired tests with swap uncertainty).

### 4.2 Example: Problem A-00

Father Youssef (type A), mother Samantha (unknown), child Lyn in North Wumponia. Query: Lyn's distribution.

Youssef has type A, so genotype AA or AO. With priors P(AA)=0.25, P(AO)=0.25, both equally likely given type A. He passes A with probability 0.75, O with 0.25.

Samantha follows population priors. Computing all inheritance combinations:

| Blood Type | Probability |
|------------|-------------|
| O          | 0.0625      |
| A          | 0.6875      |
| B          | 0.0625      |
| AB         | 0.1875      |

The system produces these exact values.

### 4.3 Results Summary

All problems across categories A-D are solved correctly. Variable elimination completes in 1-2 seconds per problem. The exact inference provides confidence that the model is correctly specified.

## 5 Discussion

### 5.1 Causal vs Diagnostic Modeling

The most important insight: model structure should follow causal direction (genotype→phenotype, parent→child) even when inference goes the opposite way. This yields:

- **Natural CPTs**: "What allele does an AO parent pass?" has an obvious answer. The reverse requires Bayes' rule.
- **Modularity**: Inheritance rules, phenotype mapping, and test semantics stay separate.
- **Flexibility**: Any query works without restructuring.

### 5.2 Intermediate Nodes

Contribution nodes simplify CPTs and correspond to biological reality (the actual allele transmitted). This aids both understanding and debugging.

### 5.3 Limitations

The current system assumes single-region families, handles only ABO types, treats standard tests as perfect, and cannot represent mutations. Extensions addressing these would be straightforward within the Bayesian network framework.

## 6 Conclusion

This work demonstrated that Bayesian networks effectively model blood type inference in family trees. The causal modeling approach—edges following biological causation rather than inference direction—proved essential for clean, modular probability specifications. The framework naturally accommodates various evidence types and correctly propagates information through complex family structures.

The broader lesson: when building probabilistic models for diagnostic inference (determining causes from effects), structure the model causally and let the inference algorithm handle the reversal.

## References

[1] Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann.

[2] Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models*. MIT Press.

[3] Dean, L. (2005). Blood Groups and Red Cell Antigens. NCBI Bookshelf.

[4] Ankan, A. & Panda, A. (2015). pgmpy: Probabilistic graphical models using Python. *SciPy Conference*.
