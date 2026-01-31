# Probabilistic Blood Type Inference Using Bayesian Networks

## Abstract

This report explores the challenge of inferring blood type probabilities in family trees when only partial information is available. We develop a Bayesian network model that represents genetic inheritance causally—from parental alleles to child alleles to observable blood types. The model handles three evidence types: direct tests, mixed-sample tests, and paired tests with potential labeling errors. Through evaluation on 29 test cases of varying complexity, we demonstrate that the causal modeling approach produces correct probability distributions while keeping the model structure clean and modular. A key finding is that modeling in the causal direction (genotype→phenotype) yields simpler probability specifications than the diagnostic direction, even when the inference goal is to determine causes from observed effects.

## 1 Introduction

Agents often need to reason about uncertain information. In medical genetics, a physician might know some family members' blood types but need to infer others. The ABO blood group system follows clear inheritance rules, yet practical inference must handle incomplete observations and imperfect laboratory tests.

**Research question:** How can we build a probabilistic model that accurately infers blood type distributions given partial family and test information?

**Contribution:** There are three key contributions in this report:
1. A Bayesian network architecture that separates genetic state (alleles) from observable phenotype (blood type)
2. Methods for incorporating three different evidence types into a unified framework
3. An evaluation comparing model accuracy across problem categories of increasing complexity

**Overview:** Section 2 describes the problem domain. Section 3 presents the Bayesian network model architecture. Section 4 evaluates the model on test cases. Section 5 discusses insights and limitations, and Section 6 concludes.

## 2 Problem Description

### 2.1 The ABO Blood Group System

Each person inherits two alleles of the ABO gene, one from each parent. There are three possible alleles: A, B, and O. The alleles combine to produce four observable blood types according to dominance rules (A and B are codominant; O is recessive).

```
┌─────────────────────────────────────────┐
│     Genotype → Blood Type Mapping       │
├──────────────┬──────────────────────────┤
│  Genotype    │  Blood Type              │
├──────────────┼──────────────────────────┤
│  AA or AO    │  Type A                  │
│  BB or BO    │  Type B                  │
│  AB          │  Type AB                 │
│  OO          │  Type O                  │
└──────────────┴──────────────────────────┘
        Figure 1: Genotype-phenotype mapping
```

This mapping creates an inference challenge: observing blood type A tells us the genotype is AA or AO, but does not distinguish between them.

### 2.2 Population Genetics in Wumponia

The scenario involves families in Wumponia, a fictional country with two regions having different allele frequencies:

```
┌─────────────────────────────────────────────────────┐
│        Allele Frequencies by Region                 │
├─────────────────┬─────────┬─────────┬───────────────┤
│  Region         │  P(A)   │  P(B)   │  P(O)         │
├─────────────────┼─────────┼─────────┼───────────────┤
│  North Wumponia │  0.50   │  0.25   │  0.25         │
│  South Wumponia │  0.15   │  0.55   │  0.30         │
└─────────────────┴─────────┴─────────┴───────────────┘
        Figure 2: Population allele frequencies
```

These frequencies serve as prior probabilities for **founder individuals**—those without known parents in the family tree.

### 2.3 Evidence Types

Three types of laboratory tests provide evidence:

1. **Standard blood type test:** Directly observes a person's blood type (always correct)
2. **Mixed blood test:** Blood from two people is combined; the result shows which antigens are present in the mixture
3. **Paired blood test:** Two people are tested together, but labels may be swapped with 20% probability

## 3 Model Architecture

### 3.1 Network Structure

The central design decision is to model **causal** relationships rather than diagnostic ones. Although we want to infer blood types from test results, we structure the network with edges pointing from causes to effects.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Bayesian Network Structure                    │
│                                                                  │
│   Father_Allele1 ──┐                    Mother_Allele1 ──┐      │
│                    ├─► Father_Contrib                    ├─► Mother_Contrib
│   Father_Allele2 ──┘         │          Mother_Allele2 ──┘         │
│                              │                                     │
│                              ▼                                     ▼
│                        Child_Allele1                        Child_Allele2
│                              │                                     │
│                              └──────────────┬──────────────────────┘
│                                             │
│                                             ▼
│                                      Child_BloodType
│                                             │
│                                             ▼
│                                       Test_Result
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
        Figure 3: Network structure for a minimal family
```

Each person is represented by three nodes:
- **Allele1, Allele2:** The two inherited alleles (values: A, B, O)
- **BloodType:** The observable phenotype (values: A, B, AB, O)

### 3.2 Inheritance Modeling

I introduce intermediate **contribution nodes** to represent the allele each parent passes to the child. The conditional probability table for contributions encodes Mendelian inheritance:

```
┌────────────────────────────────────────────────────────┐
│     Contribution CPT (Parent with alleles X, Y)        │
├────────────────────┬───────────────────────────────────┤
│  Parent Genotype   │  P(contribute X)  P(contribute Y) │
├────────────────────┼───────────────────────────────────┤
│  X = Y (e.g., AA)  │       1.0              0.0        │
│  X ≠ Y (e.g., AO)  │       0.5              0.5        │
└────────────────────┴───────────────────────────────────┘
        Figure 4: Conditional probability table for inheritance
```

### 3.3 Evidence Integration

**Mixed tests** are modeled with an additional node whose CPT encodes antigen combination:

```
┌──────────────────────────────────────────────────┐
│      Mixed Blood Test Result Logic               │
├──────────────────────────────────────────────────┤
│  If either person has A antigen → mixture has A  │
│  If either person has B antigen → mixture has B  │
│  Example: Person1=A, Person2=B → Mixture=AB      │
└──────────────────────────────────────────────────┘
        Figure 5: Mixed test semantics
```

**Paired tests** require modeling the correlation between two results. I use a joint node with 16 states (all pairs of blood types):

```
┌────────────────────────────────────────────────────────────┐
│           Paired Test CPT                                  │
├────────────────────────────────────────────────────────────┤
│  If actual types differ:                                   │
│    80% probability: reports match actual types             │
│    20% probability: reports are swapped                    │
│  If actual types are same:                                 │
│    100% probability: reports match (swap has no effect)    │
└────────────────────────────────────────────────────────────┘
        Figure 6: Paired test probability model
```

## 4 Evaluation

### 4.1 Test Categories

The evaluation uses 29 test cases organized into four categories:

```
┌────────────────────────────────────────────────────────────────────┐
│                    Problem Categories                              │
├──────────┬─────────────────────┬───────────────────┬───────────────┤
│ Category │ Family Structure    │ Evidence Types    │ # Problems    │
├──────────┼─────────────────────┼───────────────────┼───────────────┤
│    A     │ Minimal (3 people)  │ Standard only     │     11        │
│    B     │ Extended families   │ Standard only     │      6        │
│    C     │ Extended families   │ Standard + Mixed  │      6        │
│    D     │ Extended families   │ Standard + Paired │      6        │
└──────────┴─────────────────────┴───────────────────┴───────────────┘
        Figure 7: Overview of test categories
```

### 4.2 Running Example: Problem A-00

Consider a family in North Wumponia: father Youssef, mother Samantha, child Lyn. Evidence: Youssef has blood type A. Query: What is Lyn's blood type distribution?

**Step 1: Infer Youssef's genotype distribution**

Since Youssef has type A, his genotype is AA or AO. Using Bayes' rule with population priors:
- P(AA | type A) = P(type A | AA) × P(AA) / P(type A) = 1 × 0.25 / 0.5 = 0.5
- P(AO | type A) = 0.5

**Step 2: Compute allele contribution probabilities**

From Youssef: P(contribute A) = 0.5×1.0 + 0.5×0.5 = 0.75, P(contribute O) = 0.25

From Samantha (no evidence): follows population priors directly

**Step 3: Compute Lyn's distribution**

```
┌───────────────────────────────────────────────────────┐
│        Lyn's Blood Type Distribution                  │
├──────────────┬────────────────────────────────────────┤
│  Blood Type  │  Probability                           │
├──────────────┼────────────────────────────────────────┤
│      O       │  0.0625                                │
│      A       │  0.6875                                │
│      B       │  0.0625                                │
│      AB      │  0.1875                                │
└──────────────┴────────────────────────────────────────┘
        Figure 8: Solution for Problem A-00
```

The system produces exactly these values.

### 4.3 Results Summary

```
┌────────────────────────────────────────────────────────────────────┐
│                    Evaluation Results                              │
├──────────┬─────────────┬───────────────┬───────────────────────────┤
│ Category │ # Problems  │ # Correct     │ Avg. Runtime (seconds)    │
├──────────┼─────────────┼───────────────┼───────────────────────────┤
│    A     │     11      │     11        │        1.2                │
│    B     │      6      │      6        │        1.5                │
│    C     │      6      │      6        │        1.8                │
│    D     │      6      │      6        │        2.1                │
├──────────┼─────────────┼───────────────┼───────────────────────────┤
│  Total   │     29      │     29        │        1.6 (avg)          │
└──────────┴─────────────┴───────────────┴───────────────────────────┘
        Figure 9: Evaluation results by category
```

All 29 problems are solved correctly. Variable elimination provides exact inference, so matching results confirms the model is correctly specified.

## 5 Discussion

### 5.1 Causal vs Diagnostic Modeling

The most significant insight is that **causal modeling** (edges from genotype to phenotype) produces cleaner specifications than diagnostic modeling (edges from observations to conclusions), even when inference goes "backwards."

```
┌────────────────────────────────────────────────────────────────────┐
│           Comparison: Causal vs Diagnostic Modeling                │
├─────────────────────┬──────────────────────────────────────────────┤
│  Aspect             │  Causal Approach       │  Diagnostic Approach│
├─────────────────────┼────────────────────────┼─────────────────────┤
│  CPT specification  │  Natural, intuitive    │  Requires Bayes'    │
│                     │                        │  rule, prior-       │
│                     │                        │  dependent          │
├─────────────────────┼────────────────────────┼─────────────────────┤
│  Modularity         │  Components separate   │  Components tangled │
├─────────────────────┼────────────────────────┼─────────────────────┤
│  Query flexibility  │  Any query supported   │  Restructuring may  │
│                     │                        │  be needed          │
└─────────────────────┴────────────────────────┴─────────────────────┘
        Figure 10: Comparison of modeling approaches
```

### 5.2 Role of Intermediate Nodes

The contribution nodes serve two purposes:
1. **Simplify CPTs:** Without them, child allele CPTs would have 9 columns (3×3 parent allele combinations). With them, we have smaller, cleaner tables.
2. **Match biology:** They represent the actual allele transmitted during reproduction—a real quantity even if unobservable.

### 5.3 Limitations

The current implementation has several limitations:

- **Single-region assumption:** All founders assumed from same region
- **ABO only:** Real blood typing includes Rh factor and minor antigens
- **Perfect standard tests:** Only paired tests model measurement error
- **No mutations:** Cannot represent de novo mutations

## 6 Conclusion

This report demonstrated that Bayesian networks effectively model blood type inference in family trees with incomplete information. The key finding is that **causal structure**—edges following biological causation rather than inference direction—yields cleaner, more modular models.

The approach successfully handles three evidence types (standard, mixed, paired tests) and produces correct probability distributions across all 29 test cases. The broader lesson: when building probabilistic models for diagnostic inference, structure the model causally and let the inference algorithm handle the reversal.

## References

[1] Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann.

[2] Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.

[3] Dean, L. (2005). Blood Groups and Red Cell Antigens. National Center for Biotechnology Information. https://www.ncbi.nlm.nih.gov/books/NBK2267/

[4] Ankan, A. & Panda, A. (2015). pgmpy: Probabilistic graphical models using Python. *Proceedings of the 14th Python in Science Conference*.

## A Extra Comments

1. The format of problem files (JSON) is not discussed because it presents no interesting challenges—the parsing is straightforward.

2. Implementation details like function names and class structure are omitted because they are "don't-care choices" that do not affect the solution's correctness or efficiency.

3. The choice of Python and pgmpy library represents a reasonable engineering decision but is not the focus of this report—the conceptual model architecture is what matters.

4. The evaluation focuses on correctness rather than runtime because for these problem sizes, any reasonable implementation completes quickly.
