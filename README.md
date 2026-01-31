# Repository for ss25.2.1/team603

This repository contains the solution for:

**🧪 Topic:** SS25 Assignment 2.1 — Compute Blood Types using Bayesian Inference



## 📦 Requirements

- Python 3.8+
- [`pgmpy`](https://pgmpy.org/)

### 📥 Install Dependencies

```bash
# Install the pgmpy library for Bayesian Network modeling
pip install pgmpy
```



## ▶️ Usage + 📂 Input + 📤 Output (Unified Section)

### 🔧 How to Run

Run the script by passing a folder containing your input JSON files:

```bash
# Replace <input_folder> with the path to your input files
python bay.py <input_folder>
```

The script will:
- Read files matching: `problem-<category>-<number>.json`
- Generate solutions into: `<input_folder>/solution_pgmpy_full/`



### 📄 Example Input File (`problem-basic-1.json`)

```json
{
  "country": "North Wumponia",
  "family-tree": [
    { "subject": "John", "relation": "father", "object": "Alice" },
    { "subject": "Mary", "relation": "mother", "object": "Alice" }
  ],
  "test-results": [
    { "type": "bloodtype-test", "person": "John", "result": "A" },
    { "type": "bloodtype-test", "person": "Mary", "result": "O" }
  ],
  "queries": [
    { "person": "Alice" }
  ]
}
```



### 📤 Example Output File (`solution-basic-1.json`)

```json
[
  {
    "type": "bloodtype",
    "person": "Alice",
    "distribution": {
      "O": 0.25,
      "A": 0.75,
      "B": 0.00,
      "AB": 0.00
    }
  }
]
```



## 🧠 Notes

- Allele priors are based on region (North/South Wumponia)

