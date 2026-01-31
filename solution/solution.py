import sys
import os
import json
from itertools import product
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# --- Config ———
ALLELE_PRIORS = {
    "North Wumponia": {"A": 0.5, "B": 0.25, "O": 0.25},
    "South Wumponia": {"A": 0.15, "B": 0.55, "O": 0.30},
}
BLOOD_TYPES = ["O", "A", "B", "AB"]
ALLELES = ["A", "B", "O"]

# Map unordered allele pair to phenotype
def genotype_to_blood(a1, a2):
    hasA = (a1 == "A" or a2 == "A")
    hasB = (a1 == "B" or a2 == "B")
    if not hasA and not hasB:
        return "O"
    if hasA and not hasB:
        return "A"
    if not hasA and hasB:
        return "B"
    return "AB"

# Helper for mixed blood type calculation
def get_mixed_blood_type(bt1, bt2):
    hasA = ("A" in bt1) or ("A" in bt2)
    hasB = ("B" in bt1) or ("B" in bt2)
    if not hasA and not hasB:
        return "O"
    if hasA and not hasB:
        return "A"
    if not hasA and hasB:
        return "B"
    return "AB"

def solve_with_pgmpy(problem):
    country = problem.get("country")
    tests = problem.get("test-results", [])
    queries = [q.get("person") for q in problem.get("queries", [])]

    # Collect all unique people involved
    people = set()
    for r in problem.get("family-tree", []):
        people.add(r["subject"])
        people.add(r["object"])
    for t in tests:
        if t["type"] == "bloodtype-test":
            people.add(t["person"])
        elif t["type"] in ["mixed-bloodtype-test", "pair-bloodtype-test"]:
            people.add(t["person-1"])
            people.add(t["person-2"])
    for q in problem.get("queries", []):
        people.add(q["person"])

    model = DiscreteBayesianNetwork()
    evidence = {}

    # Topological sort to ensure parents' CPDs are defined before children's
    parents_map = {}
    for r in problem.get("family-tree", []):
        child = r.get("object")
        role = r.get("relation", "").split("-")[0] # "father" or "mother"
        parents_map.setdefault(child, {})[role] = r.get("subject")
    
    indegree = {p: 0 for p in people}
    children_of = {p: [] for p in people}
    for child, ps in parents_map.items():
        for role in ("father", "mother"):
            parent = ps.get(role)
            if parent:
                # Ensure parent is in people set if they are not already
                # (e.g. if they are only a parent and not in queries/tests/children of another person)
                people.add(parent) 
                children_of.setdefault(parent, []).append(child)
                indegree[child] = indegree.get(child, 0) + 1
    
    # Re-initialize indegree with potentially newly added parents
    indegree = {p: indegree.get(p, 0) for p in people}

    queue = [p for p, d in indegree.items() if d == 0]
    ordered_people = []
    
    # This loop is essentially a BFS for topological sort
    while queue:
        u = queue.pop(0)
        ordered_people.append(u)
        for v in children_of.get(u, []):
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    
    # 1. Add nodes for each person's two alleles and their blood type
    for person in people:
        model.add_node(f"{person}_Allele1")
        model.add_node(f"{person}_Allele2")
        model.add_node(f"{person}_BloodType")
        model.add_edge(f"{person}_Allele1", f"{person}_BloodType")
        model.add_edge(f"{person}_Allele2", f"{person}_BloodType")

    # 2. Define CPDs for Alleles and Blood Types based on topological order
    for person in ordered_people:
        father = parents_map.get(person, {}).get("father")
        mother = parents_map.get(person, {}).get("mother")

        # BloodType CPD: Depends on Person_Allele1 and Person_Allele2
        evidence_allele_combinations = list(product(ALLELES, ALLELES))
        blood_type_values_matrix = [[0.0 for _ in range(len(evidence_allele_combinations))] for _ in range(len(BLOOD_TYPES))]

        for col_idx, (a1_ev, a2_ev) in enumerate(evidence_allele_combinations):
            derived_blood_type = genotype_to_blood(a1_ev, a2_ev)
            for row_idx, blood_type_state in enumerate(BLOOD_TYPES):
                if blood_type_state == derived_blood_type:
                    blood_type_values_matrix[row_idx][col_idx] = 1.0

        cpd_bloodtype = TabularCPD(
            variable=f"{person}_BloodType",
            variable_card=len(BLOOD_TYPES),
            values=blood_type_values_matrix,
            evidence=[f"{person}_Allele1", f"{person}_Allele2"],
            evidence_card=[len(ALLELES), len(ALLELES)],
            state_names={
                f"{person}_BloodType": BLOOD_TYPES,
                f"{person}_Allele1": ALLELES,
                f"{person}_Allele2": ALLELES
            }
        )
        model.add_cpds(cpd_bloodtype)

        # Define CPD for Person_Allele1
        if father:
            father_contrib_node = f"{father}_Contribution_to_{person}_Allele1"
            # Add node and edges if not already added
            if father_contrib_node not in model.nodes():
                model.add_node(father_contrib_node)
                model.add_edge(f"{father}_Allele1", father_contrib_node)
                model.add_edge(f"{father}_Allele2", father_contrib_node)

            # CPD for father's contribution (probability of contributing A1 or A2)
            father_contrib_values_matrix = [[0.0 for _ in range(len(evidence_allele_combinations))] for _ in range(len(ALLELES))]
            for col_idx, (fa1_ev, fa2_ev) in enumerate(evidence_allele_combinations):
                for row_idx, contributed_allele_state in enumerate(ALLELES):
                    prob = 0.0
                    # If father's alleles are the same, he contributes that allele with prob 1.0
                    if fa1_ev == fa2_ev and contributed_allele_state == fa1_ev:
                        prob = 1.0
                    # If father's alleles are different, he contributes each with prob 0.5
                    elif fa1_ev != fa2_ev and (contributed_allele_state == fa1_ev or contributed_allele_state == fa2_ev):
                        prob = 0.5
                    father_contrib_values_matrix[row_idx][col_idx] = prob
            
            cpd_father_contrib = TabularCPD(
                variable=father_contrib_node,
                variable_card=len(ALLELES),
                values=father_contrib_values_matrix,
                evidence=[f"{father}_Allele1", f"{father}_Allele2"],
                evidence_card=[len(ALLELES), len(ALLELES)],
                state_names={
                    father_contrib_node: ALLELES,
                    f"{father}_Allele1": ALLELES,
                    f"{father}_Allele2": ALLELES
                }
            )
            model.add_cpds(cpd_father_contrib)

            # Link father's contribution to person's Allele1
            model.add_edge(father_contrib_node, f"{person}_Allele1")
            cpd_person_allele1 = TabularCPD(
                variable=f"{person}_Allele1",
                variable_card=len(ALLELES),
                values=[[1.0 if current_allele == parent_contrib_allele else 0.0 for parent_contrib_allele in ALLELES] for current_allele in ALLELES],
                evidence=[father_contrib_node],
                evidence_card=[len(ALLELES)],
                state_names={
                    f"{person}_Allele1": ALLELES,
                    father_contrib_node: ALLELES
                }
            )
            model.add_cpds(cpd_person_allele1)
        else: # No father, Allele1 is a founder allele
            prior_dist = ALLELE_PRIORS.get(country, {})
            cpd_allele1 = TabularCPD(
                variable=f"{person}_Allele1",
                variable_card=len(ALLELES),
                values=[[prior_dist.get(a, 0.0)] for a in ALLELES],
                state_names={f"{person}_Allele1": ALLELES}
            )
            model.add_cpds(cpd_allele1)

        # Define CPD for Person_Allele2
        if mother:
            mother_contrib_node = f"{mother}_Contribution_to_{person}_Allele2"
            # Add node and edges if not already added
            if mother_contrib_node not in model.nodes():
                model.add_node(mother_contrib_node)
                model.add_edge(f"{mother}_Allele1", mother_contrib_node)
                model.add_edge(f"{mother}_Allele2", mother_contrib_node)

            # CPD for mother's contribution (probability of contributing A1 or A2)
            mother_contrib_values_matrix = [[0.0 for _ in range(len(evidence_allele_combinations))] for _ in range(len(ALLELES))]
            for col_idx, (ma1_ev, ma2_ev) in enumerate(evidence_allele_combinations):
                for row_idx, contributed_allele_state in enumerate(ALLELES):
                    prob = 0.0
                    # If mother's alleles are the same, she contributes that allele with prob 1.0
                    if ma1_ev == ma2_ev and contributed_allele_state == ma1_ev:
                        prob = 1.0
                    # If mother's alleles are different, she contributes each with prob 0.5
                    elif ma1_ev != ma2_ev and (contributed_allele_state == ma1_ev or contributed_allele_state == ma2_ev):
                        prob = 0.5
                    mother_contrib_values_matrix[row_idx][col_idx] = prob

            cpd_mother_contrib = TabularCPD(
                variable=mother_contrib_node,
                variable_card=len(ALLELES),
                values=mother_contrib_values_matrix,
                evidence=[f"{mother}_Allele1", f"{mother}_Allele2"],
                evidence_card=[len(ALLELES), len(ALLELES)],
                state_names={
                    mother_contrib_node: ALLELES,
                    f"{mother}_Allele1": ALLELES,
                    f"{mother}_Allele2": ALLELES
                }
            )
            model.add_cpds(cpd_mother_contrib)

            # Link mother's contribution to person's Allele2
            model.add_edge(mother_contrib_node, f"{person}_Allele2")
            cpd_person_allele2 = TabularCPD(
                variable=f"{person}_Allele2",
                variable_card=len(ALLELES),
                values=[[1.0 if current_allele == parent_contrib_allele else 0.0 for parent_contrib_allele in ALLELES] for current_allele in ALLELES],
                evidence=[mother_contrib_node],
                evidence_card=[len(ALLELES)],
                state_names={
                    f"{person}_Allele2": ALLELES,
                    mother_contrib_node: ALLELES
                }
            )
            model.add_cpds(cpd_person_allele2)
        else: # No mother, Allele2 is a founder allele
            prior_dist = ALLELE_PRIORS.get(country, {})
            cpd_allele2 = TabularCPD(
                variable=f"{person}_Allele2",
                variable_card=len(ALLELES),
                values=[[prior_dist.get(a, 0.0)] for a in ALLELES],
                state_names={f"{person}_Allele2": ALLELES}
            )
            model.add_cpds(cpd_allele2)
            
    # 3. Handle test results as evidence
    test_counter = 0 # Initialize a counter for unique test IDs if 'id' is missing
    for t in tests:
        test_id_val = t.get('id', f"generated_test_{test_counter}") # Robust ID retrieval
        test_counter += 1

        if t["type"] == "bloodtype-test":
            person_tested = t["person"]
            result = t["result"]
            evidence[f"{person_tested}_BloodType"] = result
        
        elif t["type"] == "mixed-bloodtype-test":
            person1 = t["person-1"]
            person2 = t["person-2"]
            result = t["result"]
            test_node_name = f"MixedTest_{test_id_val}" 

            model.add_node(test_node_name)
            model.add_edge(f"{person1}_BloodType", test_node_name)
            model.add_edge(f"{person2}_BloodType", test_node_name)

            # CPT for mixed blood type
            p1_p2_bt_combinations = list(product(BLOOD_TYPES, BLOOD_TYPES))
            mixed_blood_values_matrix = [[0.0 for _ in range(len(p1_p2_bt_combinations))] for _ in range(len(BLOOD_TYPES))]
            
            for col_idx, (p1_bt_ev, p2_bt_ev) in enumerate(p1_p2_bt_combinations):
                actual_mixed_bt = get_mixed_blood_type(p1_bt_ev, p2_bt_ev)
                for row_idx, mixed_bt_state in enumerate(BLOOD_TYPES):
                    if mixed_bt_state == actual_mixed_bt:
                        mixed_blood_values_matrix[row_idx][col_idx] = 1.0

            cpd_mixed_test = TabularCPD(
                variable=test_node_name,
                variable_card=len(BLOOD_TYPES),
                values=mixed_blood_values_matrix,
                evidence=[f"{person1}_BloodType", f"{person2}_BloodType"],
                evidence_card=[len(BLOOD_TYPES), len(BLOOD_TYPES)],
                state_names={
                    test_node_name: BLOOD_TYPES,
                    f"{person1}_BloodType": BLOOD_TYPES,
                    f"{person2}_BloodType": BLOOD_TYPES
                }
            )
            model.add_cpds(cpd_mixed_test)
            evidence[test_node_name] = result

        elif t["type"] == "pair-bloodtype-test":
            person1 = t["person-1"]
            person2 = t["person-2"]
            reported_result1 = t["result-1"]
            reported_result2 = t["result-2"]

            # Create nodes for actual blood types in this test context, linked to the general blood type nodes
            node_actual_p1 = f"Actual_Pair_P1_{test_id_val}"
            node_actual_p2 = f"Actual_Pair_P2_{test_id_val}"
            
            model.add_node(node_actual_p1)
            model.add_node(node_actual_p2)
            
            # Edges from original blood type to actual test blood type (identity mapping)
            model.add_edge(f"{person1}_BloodType", node_actual_p1)
            model.add_edge(f"{person2}_BloodType", node_actual_p2)

            # CPD for Actual_Pair_P1 (identity with Person1_BloodType)
            cpd_actual_p1 = TabularCPD(
                variable=node_actual_p1,
                variable_card=len(BLOOD_TYPES),
                values=[[1.0 if bt_val == actual_bt else 0.0 for actual_bt in BLOOD_TYPES] for bt_val in BLOOD_TYPES],
                evidence=[f"{person1}_BloodType"],
                evidence_card=[len(BLOOD_TYPES)],
                state_names={node_actual_p1: BLOOD_TYPES, f"{person1}_BloodType": BLOOD_TYPES}
            )
            model.add_cpds(cpd_actual_p1)

            # CPD for Actual_Pair_P2 (identity with Person2_BloodType)
            cpd_actual_p2 = TabularCPD(
                variable=node_actual_p2,
                variable_card=len(BLOOD_TYPES),
                values=[[1.0 if bt_val == actual_bt else 0.0 for actual_bt in BLOOD_TYPES] for bt_val in BLOOD_TYPES],
                evidence=[f"{person2}_BloodType"],
                evidence_card=[len(BLOOD_TYPES)],
                state_names={node_actual_p2: BLOOD_TYPES, f"{person2}_BloodType": BLOOD_TYPES}
            )
            model.add_cpds(cpd_actual_p2)

            # New: Single node for joint reported result
            node_joint_reported_pair_test = f"Joint_Reported_Pair_Test_{test_id_val}"
            model.add_node(node_joint_reported_pair_test)
            model.add_edge(node_actual_p1, node_joint_reported_pair_test)
            model.add_edge(node_actual_p2, node_joint_reported_pair_test)

            # States for the joint reported node: all possible pairs (R1, R2)
            joint_reported_states_list = [f"{bt1}_{bt2}" for bt1, bt2 in product(BLOOD_TYPES, BLOOD_TYPES)]
            
            actual_bt_combinations = list(product(BLOOD_TYPES, BLOOD_TYPES))
            joint_reported_values_matrix = [[0.0 for _ in range(len(actual_bt_combinations))] for _ in range(len(joint_reported_states_list))]

            for col_idx, (actual_p1_ev, actual_p2_ev) in enumerate(actual_bt_combinations):
                for row_idx, (reported_p1_state, reported_p2_state) in enumerate(product(BLOOD_TYPES, BLOOD_TYPES)):
                    prob = 0.0
                    if actual_p1_ev == actual_p2_ev:
                        if reported_p1_state == actual_p1_ev and reported_p2_state == actual_p2_ev:
                            prob = 1.0
                    else: # Actual types are different
                        if reported_p1_state == actual_p1_ev and reported_p2_state == actual_p2_ev:
                            prob = 0.8 # Correct report
                        elif reported_p1_state == actual_p2_ev and reported_p2_state == actual_p1_ev:
                            prob = 0.2 # Swapped report
                    
                    # Map the (reported_p1_state, reported_p2_state) tuple to the correct row_idx
                    current_joint_state_str = f"{reported_p1_state}_{reported_p2_state}"
                    current_row_idx = joint_reported_states_list.index(current_joint_state_str)
                    joint_reported_values_matrix[current_row_idx][col_idx] = prob
            
            cpd_joint_reported_pair_test = TabularCPD(
                variable=node_joint_reported_pair_test,
                variable_card=len(joint_reported_states_list),
                values=joint_reported_values_matrix,
                evidence=[node_actual_p1, node_actual_p2],
                evidence_card=[len(BLOOD_TYPES), len(BLOOD_TYPES)],
                state_names={
                    node_joint_reported_pair_test: joint_reported_states_list,
                    node_actual_p1: BLOOD_TYPES,
                    node_actual_p2: BLOOD_TYPES
                }
            )
            model.add_cpds(cpd_joint_reported_pair_test)
            
            # Add observed results as evidence for the joint node
            evidence[node_joint_reported_pair_test] = f"{reported_result1}_{reported_result2}"

    # Check model consistency
    try:
        model.check_model()
    except Exception as e:
        print(f"Model consistency check failed: {e}. Check CPD definitions or graph structure.")
        raise 

    # Perform inference
    infer = VariableElimination(model)
    solution = []

    for query_person in queries:
        query_result = infer.query(
            variables=[f"{query_person}_BloodType"],
            evidence=evidence
        )
        
        dist = {}
        if query_result and query_result.state_names and f"{query_person}_BloodType" in query_result.state_names:
            for i, bt_state in enumerate(query_result.state_names[f"{query_person}_BloodType"]):
                dist[bt_state] = query_result.values[i] # Removed rounding
        else:
            # If no valid distribution is found, assume 0.0 for all for now.
            for bt in BLOOD_TYPES:
                dist[bt] = 0.0
            print(f"Warning: No valid distribution found for {query_person}. Setting to 0.0 for all blood types.")

        solution.append({
            "type": "bloodtype",
            "person": query_person,
            "distribution": dist
        })
    return solution

# --- Main: folder input and file-based output ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <problems_folder>")
        sys.exit(1)
    in_folder = sys.argv[1]
    if not os.path.isdir(in_folder):
        print(f"Error: '{in_folder}' is not a directory.")
        sys.exit(1)
    out_folder = os.path.join(in_folder, "solution_pgmpy_full") # New output folder
    os.makedirs(out_folder, exist_ok=True)

    # Process all problem-*.json files
    for fname in os.listdir(in_folder):
        if not fname.startswith("problem-") or not fname.endswith(".json"):
            continue
        
        parts = fname[:-5].split("-")
        if len(parts) != 3:
            continue
        _, category, num = parts
        in_path = os.path.join(in_folder, fname)
        try:
            with open(in_path) as f:
                prob = json.load(f)
        except Exception as e:
            print(f"Skipping {fname}: invalid JSON ({e})", file=sys.stderr)
            continue
        
        print(f"Processing {fname}...")
        try:
            result = solve_with_pgmpy(prob)
            out_name = f"solution-{category}-{num}.json"
            out_path = os.path.join(out_folder, out_name)
            with open(out_path, "w") as fo:
                json.dump(result, fo, indent=2)
            print(f"Written {out_path}")
        except Exception as e:
            print(f"Error solving {fname}: {e}", file=sys.stderr)