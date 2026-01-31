# Reviews

As you (hopefully) know, we will simulate the double-blind peer review process used in academic publishing.
Concretely, that means that you will have to review three reports by other students.
This document outlines how to write a review.
It's not easy to get criticized for one's work, so please be nice and constructive.


## Structure of a review
In general, a review consists of four parts:
1. Summary
2. Critical evaluation
3. Conclusion
4. Technical notes to the author

Reviews are usually written as a plaintext file (if you really want to, you can also submit a pdf file instead).


### Part 1: Summary
Here you summarize the report very concisely (maybe 2 to 3 sentences).
This part might actually end up being the hardest to write.

Example (for inspiration):
```
    The author investigates how ... and compares the traditional ... approach with ...
    According to their analysis, the ... approach outperforms ... for larger datasets.
    However, the authors note that their new approach does not generalize easily to ...
```


### Part 2: Critical evaluation
Here you discuss the report in more detail and criticize it.
This should be as long as it needs to be.
A typical critical evaluation might be a bit less than half a page.

Example phrases (for inspiration):
```
    The introduction motivates the report very well.
    It is unclear if components ... and ... are interacting via ... or ..., which would make a big difference because ...
    The author states that "... can be safely ignored", but it isn't clear to me why that is the case.
    Overall, the evaluation is very rigorous and justifies the claim that ...  is superior. However, I believe that ... cannot be compared directly because ...
    I believe that ... would improve ... without much effort - I suggest that the author tries it for the final report.
    Algorithm ... is rather complex and I couldn't understand it. I think the description might benefit from an example.
```

Further considerations:
1. Be nice - and rather criticize the content than the author
2. Mention both good and bad things
3. Try to be constructive - the author will use your review to improve their report


### Part 3: Conclusion
Here you provide a high-level summary of your evaluation (maybe 2 to 4 sentences).
In a "real" review, you would also recommend to accept/reject the paper.

Example (for inspiration):
```
    Overall, the report is well-written and discusses a very interesting approach.
    A few improvements could make the report a lot stronger, especially using an example to illustrate ... and doing ... in the evaluation.
```


### Part 4: Technical notes
Here you list all the details that you noticed while reading the report (but that are too minor to go into the critical evaluation).
This part is directly addressed to the author.

Examples (for inspiration):
1. `Page 13: Typo ("the the algorithm")` (it's not your task to proof-read, but if you notice a typo, you can of course point it out. If you notice lots of typos, it might be better to point out a few examples and recommend a spell-checker).
2. `Page 14: You repeatedly refer to "stages" - are you referring to the three "phases" that you introduced in ...?`
3. `I didn't understand the meaning of the squares in the plot in Fig. 5` (this is useful feedback and gives the author an opportunity to try to find a simpler explanation. Also mention if you think that something was harder to understand than necessary)
4. `Section 5 is very long and doesn't really have an obvious structure - maybe it's possible to split it into subsections (e.g. "Other optimization approaches", "Improving reusability", "Reducing the memory usage")`
5. `In ... you used the median - is there a reason for that? The mean seems the more obvious choice.`


