# Probabilistic Blood Type Inference Using Bayesian Networks

## Abstract

This report explores how to infer blood type probabilities in family trees when only partial information is available. We develop a Bayesian network model that represents genetic inheritance causally, from parental alleles to child alleles to observable blood types. The model handles three evidence types including direct tests, mixed sample tests, and paired tests with potential labeling errors. We tested this approach on 29 problems of varying complexity and found that it produces correct probability distributions while keeping the model structure clean and modular. A key finding is that modeling in the causal direction yields simpler probability specifications than the diagnostic direction, even when the inference goal is to determine causes from observed effects.

## 1 Introduction

Agents often need to reason about uncertain information. In medical genetics, a physician might know some family members' blood types but need to infer others. The ABO blood group system follows clear inheritance rules, yet practical inference must handle incomplete observations and imperfect laboratory tests.

**Research question** How can we build a probabilistic model that accurately infers blood type distributions given partial family and test information?

**Contribution** There are three key contributions in this report:

1. A Bayesian network architecture that separates genetic state (alleles) from observable phenotype (blood type)
2. Methods for incorporating three different evidence types into a unified framework
3. An evaluation comparing model accuracy across problem categories of increasing complexity

**Overview** First, the problem is described in more detail in Section 2. Section 3 presents the Bayesian network model architecture. Section 4 evaluates the model on test cases. Section 5 discusses insights and limitations, and Section 6 concludes the report.

## 2 Problem Description

### 2.1 The ABO Blood Group System

Each person inherits two alleles of the ABO gene, one from each parent. There are three possible alleles, namely A, B, and O. The alleles combine to produce four observable blood types according to dominance rules where A and B are codominant while O is recessive.

Genotype determines blood type as follows. A person with genotype AA or AO has blood type A. A person with genotype BB or BO has blood type B. A person with genotype AB has blood type AB. Only a person with genotype OO has blood type O.

This mapping creates an inference challenge because observing blood type A tells us the genotype is AA or AO, but does not distinguish between them.

### 2.2 Population Genetics in Wumponia

Our scenario involves families in Wumponia, a fictional country with two regions having different allele frequencies. In North Wumponia, the allele frequencies are P(A) = 0.50, P(B) = 0.25, and P(O) = 0.25. In South Wumponia, the frequencies are quite different with P(A) = 0.15, P(B) = 0.55, and P(O) = 0.30.

These frequencies serve as prior probabilities for **founder individuals**, which are those without known parents in the family tree. For individuals whose parents are known, their allele probabilities are determined by inheritance rather than population priors.

### 2.3 Evidence Types

Three types of laboratory tests provide evidence in our problem.

One type is the **standard blood type test** which directly observes a person's blood type. We assume this test is always correct.

Another type is the **mixed blood test** where blood from two people is combined. The result shows which antigens are present in the mixture. For example, if one person has type A and the other has type B, the mixture would show type AB since both antigens are present.

The last type is the **paired blood test** where two people are tested together but the laboratory might accidentally swap the labels with 20% probability. So the result attributed to person 1 could actually be person 2's blood type and vice versa.

## 3 Model Architecture

### 3.1 Network Structure

Our central design decision was to model **causal** relationships rather than diagnostic ones. Although we want to infer blood types from test results, we structure the network with edges pointing from causes to effects.

Each person is represented by three random variables in the network. The first two are Allele1 and Allele2, representing the two inherited alleles with possible values A, B, or O. The third is BloodType representing the observable phenotype with possible values A, B, AB, or O.

Figure 1 shows the network structure for a minimal family with father, mother, and child.

```
    Father_Allele1 ────┐
                       ├──► Father_Contribution ──► Child_Allele1
    Father_Allele2 ────┘                                   │
                                                           │
    Mother_Allele1 ────┐                                   │
                       ├──► Mother_Contribution ──► Child_Allele2
    Mother_Allele2 ────┘                                   │
                                                           │
                                                           ▼
                                                    Child_BloodType
                                                           │
                                                           ▼
                                                      Test_Result

                    Figure 1: Network structure for a minimal family
```

### 3.2 Why We Chose Causal Over Diagnostic Modeling

When we first approached this problem, it seemed natural to draw edges from test results toward blood types since we observe tests and want to infer types. However, this diagnostic direction creates problems.

With causal edges, the conditional probability tables are natural and easy to specify. For example, the question "what allele does a parent with genotype AO pass?" has an obvious answer of 50% for each allele. The reverse question "if the child has allele A, what was the parent's genotype?" requires Bayes rule and depends on population priors.

Causal modeling also keeps the components separate. The inheritance rules, the genotype to phenotype mapping, and the test semantics are all independent pieces of domain knowledge. In the diagnostic direction, these would become tangled together.

### 3.3 Inheritance Modeling with Contribution Nodes

We introduce intermediate **contribution nodes** to represent the allele each parent passes to the child. This was a key design decision that simplified our probability tables.

Without contribution nodes, the child's allele would depend directly on both parent alleles, requiring a table with 9 columns for the 3x3 combinations of parent alleles. With contribution nodes, we have two smaller tables instead.

Contribution nodes encode Mendelian inheritance simply. If a parent has two identical alleles like AA, they contribute that allele with probability 1.0. If a parent has two different alleles like AO, they contribute each with probability 0.5.

### 3.4 Modeling the Three Evidence Types

For **standard tests**, we simply set the BloodType node to its observed value as evidence.

For **mixed tests**, we add a new node representing the mixture result. This node depends on both individuals' BloodType nodes. The conditional probability table is deterministic because the mixture shows A antigen if either person has A, and shows B antigen if either person has B.

For **paired tests**, we needed to model the correlation between the two results. If a swap happens, both results are swapped together. We use a joint node with 16 states representing all pairs of blood types. The table encodes that if the actual types differ, there is 80% probability that reports match actual types and 20% probability that they are swapped.

## 4 Evaluation

### 4.1 Test Categories and Results

We evaluated our model on 29 test cases organized into four categories. Category A contains 11 problems with minimal families of three people and only standard tests. Category B has 6 problems with extended families and standard tests. Category C has 6 problems with extended families and mixed tests. Category D has 6 problems with extended families and paired tests.

All 29 problems were solved correctly. Since we use variable elimination which provides exact inference, the matching results confirm that our model is correctly specified.

### 4.2 Running Example

To illustrate how our model works, consider problem A-00. We have a family in North Wumponia with father Youssef, mother Samantha, and child Lyn. The only evidence is that Youssef has blood type A. We want to find Lyn's blood type distribution.

Since Youssef has type A, his genotype is either AA or AO. Using Bayes rule with North Wumponia priors, both genotypes have probability 0.5 given the blood type observation.

From this, we can compute that Youssef contributes allele A with probability 0.75 (certain if he is AA, fifty percent if he is AO) and contributes O with probability 0.25.

Samantha has no evidence so her alleles follow the population priors directly.

Working through all the inheritance combinations, our system computes Lyn's distribution as P(O) = 0.0625, P(A) = 0.6875, P(B) = 0.0625, and P(AB) = 0.1875. This matches the expected solution exactly.

### 4.3 Comparison of Problem Difficulty

Figure 2 shows how the problem categories differ. The Category A problems are the simplest since they involve only three people and direct blood type tests. Categories B through D introduce additional challenges with larger families and more complex evidence types.

```
    Category A (11 problems): Simple families, standard tests only
         ████████████████████████████████  All correct
    
    Category B (6 problems): Extended families, standard tests
         ██████████████████████████████████  All correct
    
    Category C (6 problems): Extended families, mixed tests
         ██████████████████████████████████  All correct
    
    Category D (6 problems): Extended families, paired tests
         ██████████████████████████████████  All correct

                    Figure 2: Results across problem categories
```

Since all categories achieved perfect accuracy, we are confident that our causal modeling approach handles both simple and complex scenarios well.

## 5 Discussion

### 5.1 Key Insights

What we learned from this work is that causal modeling produces cleaner specifications than diagnostic modeling, even when the inference goal is to determine causes from effects. Causal structure works well for many types of probabilistic modeling.

Another important finding is that intermediate nodes like our contribution nodes can greatly simplify probability tables. These nodes also correspond to meaningful biological concepts, namely the actual allele transmitted during reproduction, which makes the model easier to understand and debug.

### 5.2 Design Decisions We Made

We chose to use the pgmpy library for Bayesian network construction and inference. This allowed us to focus on the modeling aspects rather than writing inference algorithms from scratch. Variable elimination in pgmpy provided exact inference which was important for validating correctness.

We represented each problem in JSON format which was simple to parse. The family tree was represented as a list of parent child relationships, and we used topological sorting to ensure parents were processed before children when building the network.

For paired tests, our initial attempt modeled the two results as separate nodes with independent noise. This failed because it did not capture the correlation that both results are swapped together or neither is swapped. Switching to a joint node fixed this problem.

### 5.3 Limitations

Our current system has several limitations. All founders are assumed to come from the same region of Wumponia. We only handle ABO blood types while real blood typing includes Rh factor and other antigens. Standard tests are assumed perfect while only paired tests model measurement error. The system cannot represent de novo mutations where a child has an allele neither parent carries.

## 6 Conclusion

In this report we have explored how to infer blood type probabilities in family trees with incomplete information. A key challenge was to find a suitable model structure. It turns out that causal modeling where edges point from genotype to phenotype is much cleaner than diagnostic modeling.

Our approach successfully handles three evidence types including standard tests, mixed sample tests, and paired tests with label swaps. The system produces correct probability distributions across all 29 test cases.

A broader lesson from this work is that when building probabilistic models for diagnostic inference, one should structure the model causally and let the inference algorithm handle computing probabilities in the reverse direction.

## References

[1] Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann.

[2] Koller, D. and Friedman, N. (2009). Probabilistic Graphical Models. MIT Press.

[3] Dean, L. (2005). Blood Groups and Red Cell Antigens. National Center for Biotechnology Information.

[4] Ankan, A. and Panda, A. (2015). pgmpy: Probabilistic graphical models using Python. Proceedings of the 14th Python in Science Conference.

## A Extra Comments

1. The format of problem files (JSON) is not discussed in detail because the parsing was simple and presented no interesting challenges.

2. Code details like function names and class structure are omitted because they are what we call "don't care choices" that do not affect the correctness of the solution.

3. We did not include runtime comparisons because for these problem sizes any reasonable code completes quickly. The focus is on correctness rather than efficiency.

4. This report does not have a separate preliminaries section because the necessary background on Bayesian networks is integrated into the model architecture discussion where it is most relevant.
