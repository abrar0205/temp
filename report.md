# Blood Type Inference Using Bayesian Networks: A Study on Genetic Probability in Wumponia

---

## Abstract

This report talks about how I built a system to figure out blood types in families when we only know some pieces of information. The idea came from a genetics problem in the fictional country of Wumponia where different regions have different allele distributions. I used Bayesian networks to model how blood types pass from parents to kids, and the results were pretty good. The system can handle simple families with just mom, dad, and child, but also works for bigger family trees with multiple generations. What makes this interesting is that even though arrows in Bayesian networks go one direction, we can still compute probabilities going backwards. That was the main thing I learned from this project.

---

## 1. Introduction

### Why This Matters

Blood types are one of those things most people don't think about until they need a blood transfusion or there's some kind of medical situation. But from a computer science point of view, they're actually a neat example of probabilistic reasoning. You have clear rules about how types get inherited, but there's still uncertainty because you don't always know both alleles a person has.

When I first looked at this problem, I thought it would be straightforward. Parents give alleles to kids, kids get a blood type, done. But it got complicated fast once I started thinking about what happens when you only know some test results and have to work backwards.

### The Research Question

The main thing I wanted to figure out was: can we build a Bayesian network that accurately predicts blood type probabilities given incomplete information about family trees and test results? And more practically, can we do this without the probability tables becoming unmanageably large?

### What I Did

I ended up building a system in Python using the pgmpy library. The core idea is to represent each person's alleles as separate nodes, then connect them through the family tree structure. Test results become evidence that constrains the possibilities. The trickiest part was handling the different test types, especially the pair tests where labels can get swapped.

### How This Report Is Organized

Section 2 covers the background stuff you need to know before diving into the solution. Section 3 is the main part where I explain how the Bayesian network is structured and why I made certain choices. Section 4 shows some examples of how it works on actual problems. Section 5 wraps things up with what I learned and what could be done differently.

---

## 2. Preliminaries

### Blood Type Genetics (The Short Version)

Everyone has two alleles that determine blood type. These can be A, B, or O. You get one from your mom and one from your dad. The combination gives your blood type:

- Two O alleles means type O
- At least one A (and no B) means type A  
- At least one B (and no A) means type B
- One A and one B means type AB

The tricky bit is that O is recessive. So someone with type A blood might have AA or AO genotype. You can't tell just from looking at the blood type.

### Population Differences in Wumponia

Wumponia has two regions with different allele frequencies:

| Region | A | B | O |
|--------|---|---|---|
| North  | 50% | 25% | 25% |
| South  | 15% | 55% | 30% |

This matters for founders (people without known parents in our family tree). Their prior probabilities come from whichever region they're from.

### Bayesian Networks Basics

A Bayesian network is basically a directed graph where nodes are random variables and edges show dependencies. Each node has a conditional probability table (CPT) that says how likely each state is given the parents' states.

The cool thing is that once you set up the network, you can compute any probability you want. Doesn't matter if the edges point "towards" or "away from" what you're trying to find. This confused me at first because my instinct was to point arrows from tests to blood types. But the arrows should follow causality, not the direction of inference.

### Variable Elimination

This is the algorithm I used for inference. It's exact (no approximation) and works well for networks that aren't too big. The basic idea is to sum out variables one at a time, keeping track of the relevant factors. It's more efficient than computing the full joint distribution.

---

## 3. The Solution

### Network Structure Overview

Let me walk through how I set up the network. For each person, I create three types of nodes:

1. **Allele1** - the allele from dad
2. **Allele2** - the allele from mom  
3. **BloodType** - the observable blood type

The blood type node depends on both allele nodes. That part is straightforward.

For inheritance, I had to think about how to model a parent passing an allele. A parent with, say, genotype AB will pass A with 50% probability and B with 50% probability. A parent with AA passes A for sure. This led me to add "contribution" nodes that sit between the parent's alleles and the child's allele.

Here's a rough picture of what it looks like for one parent-child relationship:

```
Dad_Allele1 ----+
                |----> Dad_Contribution ----> Child_Allele1
Dad_Allele2 ----+

Mom_Allele1 ----+
                |----> Mom_Contribution ----> Child_Allele2
Mom_Allele2 ----+

Child_Allele1 ----+
                   |----> Child_BloodType
Child_Allele2 ----+
```

### Why I Used Intermediate Nodes

At first I tried having the child's allele depend directly on both of the parent's alleles. That works, but the CPT has 9 columns (3 x 3 combinations of parent alleles) and each row represents a different value for the child's allele. Not too bad.

But then I realized the contribution node makes things cleaner conceptually. It captures the biological process better: the parent contributes ONE of their two alleles with equal probability. Plus it helps when debugging because you can check intermediate values.

### Handling Founder Individuals

For people without known parents, I just use the population priors. If someone is from North Wumponia, their Allele1 has P(A)=0.5, P(B)=0.25, P(O)=0.25. Same for Allele2. These are independent because we're assuming random mating within populations.

### The Three Test Types

This is where things got interesting. The assignment has three kinds of tests:

**Regular blood type test**: This is easy. You observe someone's blood type directly. It becomes evidence on their BloodType node.

**Mixed blood test**: They combine blood from two people and report what antigens are present. If either person has A, the mix shows A. If either has B, the mix shows B. So two people with types A and B would give a mixed result of AB. I modeled this as a new node that depends on both people's blood types, with a deterministic CPT.

**Pair blood test**: This was the hard one. Two people get tested, but the labels might get swapped with 20% probability. So if Alice is actually type A and Bob is actually type B, you might see the report say Alice=A, Bob=B (80% chance) or Alice=B, Bob=A (20% chance).

For pair tests, I created a joint node that represents both reported results together. The CPT considers whether a swap happened. If both people have the same blood type, there's no swap effect. If they're different, you get the 80/20 split.

### Processing Order Matters

Bayesian networks need to be acyclic, and pgmpy needs CPTs defined in order. So I had to sort people topologically based on the family tree. Parents come before children. Then when I'm defining a child's CPT, the parent nodes already exist.

I did this with a simple BFS where you start with people who have no parents in the tree (indegree zero) and work down.

### The Code Structure

The main function `solve_with_pgmpy` does these steps:

1. Parse the problem JSON to get family tree, tests, and queries
2. Collect all people involved
3. Build the network structure with all nodes and edges
4. Define CPTs in topological order
5. Add test results as evidence
6. Run variable elimination for each query
7. Return the probability distributions

One thing that took me a while to get right was making sure evidence variable names matched exactly. If you add a node called "Alice_BloodType" but then try to set evidence on "Alice_Bloodtype" (lowercase t), it silently fails. Python dicts don't complain about extra keys.

---

## 4. Results and Examples

### Simple Family Example

Let's trace through problem-a-00. We have Youssef (father) and Samantha (mother) with child Lyn. Youssef tested as type A. What's Lyn's blood type distribution?

Since Youssef has type A, his genotype is either AA or AO. Given North Wumponian priors:
- P(AA) is proportional to 0.5 × 0.5 = 0.25
- P(AO) is proportional to 0.5 × 0.25 × 2 = 0.25

So it's 50/50 whether he has a hidden O allele.

For Samantha, we know nothing except she's from North Wumponia. Her allele distribution is just the population priors.

Working through all combinations... honestly I didn't do this by hand. The point is the program outputs:

```json
{
  "O": 0.0625,
  "A": 0.6875,
  "B": 0.0625,
  "AB": 0.1875
}
```

This matches the expected solution, so the model is working correctly.

### Mixed Blood Test Example

Problem-c-00 has siblings Sasha and Agnieszka whose blood gets mixed and shows type A. The mixed test tells us neither sibling can have type B (otherwise the mix would show B too). This constrains both of their distributions.

Since they share a mother (Zeinab), their blood types aren't independent. If Sasha has genotype with B, that increases the chance Zeinab carries B, which affects Agnieszka's probability.

The system handles this correctly by modeling the mixed test as a node that depends on both blood types, with evidence that it shows A.

### Pair Test with Potential Swap

The pair tests in the D problems were trickiest to implement. Consider problem-d-00 where Lindsay and Kim get pair-tested with results B and A. But wait, labels might be swapped 20% of the time.

The model says: with 80% probability, Lindsay=B and Kim=A. With 20% probability, Lindsay=A and Kim=B. But we also know from another test that Lindsay is definitely type A. So the "swap happened" scenario (20%) is actually the correct one here.

When you work through the math, the system correctly downweights the scenarios where Lindsay has type B.

### Performance

All problems from A through D solved correctly. Run time is negligible for these family sizes - maybe a second or two per problem. The variable elimination algorithm is efficient enough that I didn't need to worry about optimization.

---

## 5. Discussion and Conclusions

### What Worked Well

The biggest win was using causal modeling from the start. I could have tried to shortcut things by having test results point directly to conclusions, but that would have made the probability calculations much messier. By following the actual causal structure (alleles determine blood type, evidence comes from observations), everything fit together naturally.

Using pgmpy was also a good call. I could have implemented Bayesian network inference from scratch, but there's no point reinventing the wheel. The library handles all the bookkeeping and lets me focus on modeling the problem correctly.

### What Was Tricky

The pair tests took me the longest to get right. My first attempt modeled the two reported results as separate nodes, but that didn't capture the correlation between them. The swap either happens or it doesn't - you can't have one swapped and one not swapped.

Switching to a single joint node for the pair test fixed this. The state space is larger (16 combinations of two blood types instead of 4 + 4), but the dependencies are modeled correctly.

### Limitations

The current implementation assumes everyone in a family tree is from the same country. If you had a family spanning both Wumponias, you'd need to specify each founder's origin separately. That wouldn't be hard to add, but the current input format doesn't support it.

Also, I only handle ABO blood types. Real genetics is way more complex - Rh factor, rare antigens, all sorts of things. But for this assignment, ABO is enough.

### What I Learned

The main takeaway for me was about the difference between causal direction and inference direction in Bayesian networks. You build the network with arrows following causality (parents cause children to have certain alleles), but you can query probabilities in any direction. The math works out the same whether you're predicting from parents to children or inferring parents from children's test results.

I also got better at reading probability tables and thinking about what each conditional probability means. Before this project, CPTs were kind of abstract. Now I can look at one and understand what biological process it represents.

### Future Ideas

If I had more time, I'd want to try:

1. Adding Rh factor - it's another independent genetic system, so it would mostly just double the number of allele nodes.

2. Handling measurement error in regular blood tests, not just pair tests. Real lab tests aren't perfect.

3. Visualization of the Bayesian network for debugging. Right now I just print things out, but a graph would be easier to check.

4. Optimizing for really large family trees. Variable elimination might struggle with hundreds of people, though I don't know where the practical limit is.

---

## References

1. pgmpy documentation: https://pgmpy.org/
2. Murphy, K. "Machine Learning: A Probabilistic Perspective" - Chapter on graphical models
3. Course materials on Bayesian networks and genetic modeling
4. Wikipedia articles on ABO blood group system and Mendelian inheritance

---

*This report was written for Assignment 2.1 of the AI Systems Project course at FAU Erlangen-Nürnberg.*
