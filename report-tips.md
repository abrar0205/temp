# Report Writing
TLDR: The report is something like a scientific paper or a mini bachelor's thesis.
This document tries to provide some more guidance.
We encourage you to use the *Writing* room on matrix.
Writing requires experience: The more you write, the faster and better you get at it.
The report can be a good preparation for your (longer) master's thesis.

There is also an **[example report](example-report.pdf)** with comments.


## Rules
* **Before you start writing** you should tell us that you want to write a report and what you write it about.
    There are some constraints what topics you may write a report about and it would be a pity if you waste a lot
    of time on a topic that is unsuitable.
* We will simulate the double-blind peer review process (see also [review-tips.md](review-tips.md)).
    After submitting a preliminary report (without your name on it!), we will
    anonymously forward it to other students, who will then write a review for you.
    A few weeks later, you will get those reviews (anonymously) and can use them to improve your report.
    Your grade will then be based on the final report.
* You cannot choose a assignment for your report if your team partner for that assignment also writes a report on it.
* The report must be written by you alone but:
    * You can ask other people for feedback and use it to improve your report (including feedback from the reviews)
    * You are encouraged to use the *Writing* room on Matrix
* Your report should be written in LaTeX. Of course it can make sense to use other tools to create graphs/figures etc.
* Your report should be submitted as a single PDF file to your admin repo (the README has instructions for the file name etc.).


## Using the *Writing* Room
It is difficult to provide a comprehensive guideline for writing a report and it would be of limited use anyway.
The *Writing* room is supposed to compensate for that by allowing you to ask any questions that might come up.
We would like to encourage you to take advantage of that.

To illustrate what kinds of questions you could ask, here are a few examples:
* Is there a better way to do XYZ in LaTeX?
* This sentence feels way too convoluted - how could I simplify it?
* Is it okay to make a separate section for XYZ?
* Do I need a table of contents?
* This figure is too big - any ideas how to make it smaller?
* For related work I only have XYZ - is there something I missed?
* I just can't get myself to start writing - any tips?
* Should I mention XYZ or is it an unnecessary detail?


## "Make it your own problem"
Your report should resemble a scientific publication rather than a homework assignment.
Therefore, you should largely write it as if the assignment itself doesn't exist and you just wanted to find out things yourself.
A very similar thing happens when you write a bachelor's or master's thesis (which this project is supposed to prepare you for):
rather than saying "my supervisor told me to do XYZ" you would say "I did XYZ to find out ..." etc.


## Thinking about your implementation as an experiment
* When you are writing about a system, you should think of this as an experiment that tells us something about the "space of all information systems", concretely in the vicinity of the system you have built.
* You may want to think about the experiment you did as a light that illuminates this region of the information system space, because from your implementation you can derive more information than just your code.
* Generally, while implementing you make lots of little and large choices. They can be classified into
    1. *don't-know choices*, where before the experiment you do not know what the "right choice" might be (you may have a gut feeling, but that will only be verified by your implementation),
    2. *don't-care choices*, - e.g. which programming language, ... - where whatever you choose is reasonable.
    The don't-care choices do not really have to be justified (they affect nothing, e.g. you could have implemented your program in Java instead of Python), but they "span the region of the space you can talk about" - your implementation is just a representative of a whole class of similar systems.
* The report is really the artefact that comes out of your research in AISysProj, not the code. 
    The code is just the experiment which justifies the findings, while the report talks about more general properties of information systems and is therefore more valuable.  

It might also help to imagine writing the report for "the younger you":
I.e. what would you have liked to know when you started out trying to solve the problem?
Try to help this person as much as possible (they only get to see your report and the references in there; not the code). 


## How long should the report be?
Well, we really do not like to prescribe a range of pages for your report. 
Instead, here are some more comments that should guide your writing:
* Your goal should be to write a well-rounded report, not *n* pages of text.
* Formatting makes a huge difference anyway.
* Writing a short report can be much more difficult than writing a long one because for a short report you have to condense a complex solution into a succinct description.
    It is similar to how writing the abstract and introduction can be the hardest part.
    When writing an academic paper, staying within the page limit is usually a major challenge.
* Picking good examples and diagrams can save a lot of space.
* We do not consider the page number when grading. In fact, in the past we had a very good 3-page report (dense formatting with two columns and a small font)
    and very good reports with well above 20 pages (and bad ones of either length as well).
* What matters is in a way completeness: Did you describe and explain your solution (not the code but the ideas behind it) in a good, understandable way?
    Did you motivate your design decisions (and ideally evaluate them)? etc.
    If you do that on 3 pages, that's great.
    If you do it on 30 pages, that's also great.
* I think a typical report is a bit longer than 10 pages.
    Making it much shorter (and still being complete) is difficult.
    Making it longer is definitely possible, especially if you did a lot of work.


## General Structure

### Abstract
The abstract is a brief summary of your work. You can think of it as an advertisement:
people would decide whether your work is worth reading based on the abstract.
Therefore, it should include a short motivation, a summary of what you did and what the results were.

### Introduction
In general, the introduction consists of the following parts:
1. Motivation: Why does your work matter? How does it fit into the big picture?
2. Research question: What were you trying to find out?
    (e.g. "Is it possible to do ... using a SAT solver?"
    "Can algorithm XYZ be applied to a 3-player game like Sternhalma?")
3. Contribution: A summary of what you did to answer the research question
    (e.g. "I came up with an architecture to separate cost functions from the search algorithm.
        I tested an implementation of that architecture on data set ... using cost functions ....
        In particular, I timed two different search algorithms: ... and ...")
4. Overview: A very brief summary of what the next sections discuss
    (e.g. "Section 2 discusses the related work and preliminaries.
        Section 3 describes a new architecture and section 4 how we applied it to ...
        In section 5 we discuss the results and section 6 concludes this report")

### Preliminaries
Here you should describe background knowledge that is needed to understand the rest of your report.
In contrast to the main part of your report, it discusses things that you didn't come up with yourself.

Examples:
* What are constraint satisfaction problems?
* What is the SAT solver MiniSat?
* How does Dijkstra's algorithm work?
* What is SPARQL?

The preliminaries can be a good place to establish terminology and notations.
Make sure that you only mention things that are relevant to your report.

### Main part
Here you describe your actual work.
That can include both theoretical and practical aspects:
* I did ... to convert ... into a SAT problem
* Doing ... made it possible compare approach 1 and 2

The structure and contents of the main part can vary a lot.

If possible, it is also a good idea to have an evaluation section,
which discusses how well your solution works, what its limits are (could it scale to larger problems?),
maybe the comparison of different approaches.
If you have numbers (e.g. problem size vs run time) to back that up, it would make the evaluation much stronger.

Tips:
* Figures and running examples don't just make it easier to understand your report, they can also help with the writing process (therefore it is a good idea to make them early, even if you want to change details later on)
* Implementation details like the names of functions and classes (or even the choice of programming language) are not particularly interesting.
    More interesting is an overview of the (conceptual) architecture or processing pipeline.
    Here it usually helps to introduce names for the different parts (e.g. "The translation works in two steps: During the **preprocessing step**,
    all inputs are transformed into ..., and during the **simplification step**, the rules described in ... are applied exhaustively").
    Usually, it is a good idea to use figures for illustration.

### Conclusion
Briefly summarize your work and try to answer the research question.
This is also a good place for future work (what you would do next, e.g. try out a different algorithm, apply your solution to a different problem, ...).


## Random tips:
* Use running examples and figures for illustration. They also help with the writing.
    A running example does not have to show the full complexity and you can of course use
    other examples to illustrate particular challenges.
* Don't write the report from the beginning to the end.
    It tends to be easier to start with the main part and the preliminaries.
* It often helps to introduce names. E.g. "Implement an agent for FAUHalma" is easier than "Implement an agent for the Sternhalma variant discussed below".
    If you introduce names, use them consistently.
* Passive voice can make sentences unnecessarily complex - consider using active voice instead.
* Make sure your sections are well-structured. If you have a long section, consider introducing sub-sections.
    Getting the section-internal structure right can be very challenging, especially, if you want to discuss a lot of different things.
    However, having a good structure is important to make the section understandable, so it is worth spending time on it.
* Some people have a habit of first explaining a complex idea and then illustrating it with an example.
    Typically, that means that the reader cannot understand the explanation until they have read the example - and then have to re-read the explanation.
    It is much better introduce an example first and then reference it through-out your explanation.
* Don't start the writing too late (you can start before you have a complete solution - for your bachelor's/master's thesis you definitely should start earlier). Reasons:
    * Writing is hard and might take longer than you expect
    * It can be difficult to write fulltime


## Most common reasons reports get a bad grade
1. Using LLMs to write the report. While LLMs can help to e.g. suggest wording, they are not good at writing reports.
   Make sure *you* are the author, not the LLM.
   Even though the LLM may produce sentences that sounds smart, they can be wrong, overly vague, irrelevant, etc.
   Furthermore, LLMs often get the style wrong, e.g. by producing long lists of bullet points instead of a coherent text.
2. Not having solved the assignment (e.g. because your team partner did all the work).
   This typically means that you don't know enough about the assignment to write a good report.
3. An overly vague report (possible as a result of 1. or 2.).
4. Just writing up a step-by-step description of your solution (in a way, like a lab report).
   Instead, you should something more like a scientific paper that focuses on the ideas behind your solution
   and evaluates them.



