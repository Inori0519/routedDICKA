import json
from fractions import Fraction
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


class ManuscriptContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = 1000
        cls.source = (PAPER / "main.tex").read_text(encoding="utf-8")
        cls.main, _, cls.appendix = cls.source.partition(r"\appendix")
        cls.generated = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((PAPER / "generated").glob("*.tex"))
        )
        cls.compiled = cls.source
        for relative in re.findall(r"\\input\{([^}]+)\}", cls.source):
            path = PAPER / relative
            if path.suffix != ".tex":
                path = path.with_suffix(".tex")
            cls.compiled += "\n" + path.read_text(encoding="utf-8")

    def test_manuscript_is_one_document_with_inline_appendices(self):
        self.assertEqual(self.source.count(r"\appendix"), 1)
        for command in (r"\documentclass", r"\begin{document}",
                        r"\end{document}", r"\bibliography{"):
            self.assertEqual(self.source.count(command), 1, msg=command)
        self.assertFalse((PAPER / "supplement.tex").exists())
        self.assertNotIn("Supplementary Information", self.source)

    def test_rate_notation_is_plain_and_bff_status_remains_explicit(self):
        self.assertNotIn(r"\underline", self.compiled)
        self.assertNotRegex(self.compiled, r"h_\*|h_\{\*\}|K_\{[^}]*guar")
        self.assertIn(r"\label{eq:dw-rate}", self.main)
        self.assertIn("H(A_0|E)", self.main)
        self.assertIn(r"\EC", self.main)
        self.assertIn("BFF", self.main)
        self.assertIn("lower bound", self.main)
        self.assertNotIn(r"\pos", self.compiled)
        self.assertNotRegex(self.compiled, r"\]_\+|\]_\{\+\}")
        self.assertIn("Nonpositive entropy differences", self.main)
        self.assertIn("positive-rate region", self.main)

    def test_main_text_follows_five_section_framework(self):
        sections = re.findall(r"^\\section\{([^}]+)\}", self.main, flags=re.MULTILINE)
        self.assertEqual(
            sections,
            [
                "Introduction",
                "Tripartite Routed Conference Key Agreement",
                "Analytical Security and Ideal Rates",
                "Key Rates and Routing Advantage",
                "Discussion",
            ],
        )

    def test_main_protocol_is_ternary_and_binary_formula_is_isolated(self):
        self.assertIn(r"b,c\in\{+1,-1,\perpout\}", self.main)
        self.assertIn(r"M^{(j)}_{\perpout|M}=(1-\eta_j)\id", self.main)
        self.assertIn(r"\EC&=1-L(\nu)\min\{\eta_B,\eta_C\}", self.main)
        self.assertNotIn(r"\left(1-\frac{\eta}{2}\right)", self.main)
        self.assertIn(r"\perpout\mapsto+1", self.appendix)

    def test_parity_scores_have_the_required_four_signs(self):
        for label in ("eq:remote-parity-score", "eq:near-score"):
            end = self.main.index(rf"\label{{{label}}}")
            start = self.main.rfind(r"\begin", 0, end)
            block = self.main[start:end]
            terms = re.findall(r"([+-])?\\langle", block.replace("\n", ""))
            signs = [term or "+" for term in terms]
            self.assertEqual(signs, ["+", "+", "-", "+"])

    def test_revision_markup_is_disabled(self):
        for marker in (r"\showrevisionstrue", r"\begin{revision}", r"\rev{"):
            self.assertNotIn(marker, self.source)
        self.assertNotRegex(self.source, r"\\(?:textcolor|color)\{red\}")

    def test_station_alphabets_and_ideal_alice_slice_are_explicit(self):
        self.assertNotIn("tab:recording-conventions", self.source)
        self.assertIn(r"\eta_A=1", self.main)
        self.assertIn(r"\eta_{B_S}=\eta_{C_S}=\eta_S", self.main)
        self.assertIn(r"\{+1,-1\}", self.main)
        self.assertIn(r"\perpout\mapsto+1", self.main)

    def test_compact_statistics_are_the_reported_secrecy_model(self):
        self.assertIn("low-dimensional coarse-grained composite statistics", self.main)
        self.assertIn("full ternary record", self.main)
        self.assertIn(r"\label{app:composite-reduction}", self.appendix)
        self.assertIn(r"\label{eq:si-composite-safety}", self.appendix)
        self.assertNotIn("complete remote-table BFF--NPA formulation", self.main)
        for label in (
            "app:full-ternary-model",
            "eq:si-complete-behavior-constraints",
            "tab:si-complete-models",
        ):
            self.assertNotIn(rf"\label{{{label}}}", self.appendix)

    def test_compact_maps_remain_defined_for_reproducibility(self):
        equations = re.sub(r"\s+", "", self.appendix.replace("&", ""))
        for expression in (
            r"S_j^+=B^S_{+|j}C^S_{+|0}",
            r"R_j^+=\overline B_{+|j}\overline C_+",
            r"L_+=B^L_{+|3}C^L_{+|0}+B^L_{-|3}C^L_{-|0}",
            r"L_-=B^L_{+|3}C^L_{-|0}+B^L_{-|3}C^L_{+|0}",
            r"L_\perp=\id-L_+-L_-",
            r"\overline B_{+|j}=B^L_{+|j}+B^L_{\perpout|j}",
        ):
            self.assertIn(re.sub(r"\s+", "", expression), equations)

    def test_readme_identifies_the_completed_revision_baseline(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("low-dimensional", readme)
        self.assertIn("full ternary record", readme)
        self.assertIn("full-ternary scanning implementation is", readme)
        self.assertIn("not included here", readme)
        self.assertFalse("historical composite-projection study" in readme)
        for path in (ROOT / "README.md", PAPER / "README.md", PAPER / "PROOF_DATA.md"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("paper/supplement.tex", text)
            self.assertNotIn("`supplement.tex`", text)

    def test_recording_control_preserves_the_compact_physical_comparison(self):
        recording = self.appendix.split(r"\label{app:recording}", 1)[1]
        recording = re.split(r"\\(?:sub)?section\*?\{", recording, maxsplit=1)[0]
        prose = " ".join(recording.split())
        self.assertRegex(prose, r"(?i)tilted[^.]*binary[- ]project")
        self.assertIn("0.85", prose)
        self.assertRegex(prose, r"0\.0366\d*")
        self.assertRegex(prose, r"0\.1938\d*")
        self.assertRegex(prose, r"(?i)decod|reconciliation")
        self.assertRegex(prose, r"(?i)phase[^.]*eras|eras[^.]*phase")
        self.assertRegex(prose, r"(?i)all (?:attempted )?key rounds|no postselection")
        self.assertNotIn(r"\begin{table}", recording)

    def test_required_appendix_proofs_are_present(self):
        for label in (
            "app:ternary", "app:ideal", "app:srq", "app:bff", "app:attack",
            "app:phase-local", "app:recording", "app:ideal-mwrf",
            "app:composite-reduction", "app:model-inventory", "app:remote-bell-witness",
        ):
            self.assertIn(rf"\label{{{label}}}", self.appendix)

    def test_appendices_are_compressed_to_at_most_a_through_e(self):
        sections = re.findall(r"^\\section\{([^}]+)\}", self.appendix, re.MULTILINE)
        self.assertGreater(len(sections), 0)
        self.assertLessEqual(len(sections), 5, msg="Appendices must stop at E")
        roots = re.findall(
            r"^\\section\{[^}]+\}\s*\\label\{([^}]+)\}",
            self.appendix, re.MULTILINE,
        )
        self.assertEqual(roots, [
            "app:ternary", "app:srq", "app:ideal-mwrf", "app:bff", "app:attack",
        ])
        recording = self.appendix.index(r"\label{app:recording}")
        self.assertLess(self.appendix.index(r"\label{app:ideal-mwrf}"), recording)
        self.assertLess(recording, self.appendix.index(r"\label{app:bff}"))
        for label in (
            "app:comparison", "app:numerics", "app:reproduction",
            "tab:si-mwrf-numerics", "tab:representative-point", "tab:si-radau",
            "tab:si-recording", "eq:si-certificate-identity",
        ):
            with self.subTest(label=label):
                self.assertNotIn(rf"\label{{{label}}}", self.compiled)

    def test_manuscript_has_no_file_manifest_or_audit_narrative(self):
        body = self.source.partition(r"\begin{document}")[2]
        self.assertNotIn(r"\path", self.source)
        self.assertNotRegex(body, r"source(?:_|\\_)data|verification/|outputs/")
        self.assertNotRegex(
            body,
            r"(?i)\b(?:electronic|checkers?|manifests?|files?|reproduction|reproducibility)\b",
        )
        headings = re.findall(r"\\(?:sub)*section\*?\{([^}]+)\}", body)
        for heading in headings:
            self.assertNotRegex(heading, r"(?i)audit|exact checks|reproduc|proof data")

    def test_code_availability_is_one_sentence_with_only_the_repository_url(self):
        self.assertNotIn(r"\section*{Data availability}", self.source)
        marker = r"\section*{Code availability}"
        self.assertEqual(self.main.count(marker), 1)
        availability = self.main.split(marker, 1)[1].split(r"\section", 1)[0].strip()
        urls = re.findall(r"https?://[^}\s]+", self.source)
        repository = "https://github.com/brant-fudan/routed-dicka"
        self.assertEqual(urls, [repository])
        self.assertEqual(re.findall(r"\\(?:url|href)\{([^}]+)\}", self.source), [repository])
        self.assertIn(repository, availability)
        sentence = re.sub(r"\\(?:url|href)\{[^}]+\}", "repository", availability)
        self.assertEqual(len(re.findall(r"[.!?](?:\s|$)", sentence)), 1)
        self.assertTrue(sentence.endswith("."))
        self.assertLessEqual(len(sentence.split()), 35)
        self.assertNotRegex(availability, r"(?i)\bpublic(?:ly)?\b|private|accessible|commit")

    def test_all_internal_references_resolve_without_compiling(self):
        label_list = re.findall(r"\\label\{([^}]+)\}", self.compiled)
        labels = set(label_list)
        self.assertEqual(len(label_list), len(labels), msg="Duplicate manuscript labels")
        refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", self.compiled))
        self.assertEqual(refs - labels, set(), msg="Unresolved manuscript references")

    def test_all_citation_keys_exist(self):
        bibliography = (PAPER / "references.bib").read_text(encoding="utf-8")
        defined = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))
        used = set()
        for source in (self.compiled, self.generated):
            for group in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", source):
                used.update(key.strip() for key in group.split(","))
        self.assertEqual(used - defined, set(), msg="Undefined bibliography keys")

    def test_latex_environments_are_balanced(self):
        for name, source in (("manuscript", self.source), ("generated", self.generated)):
            stack = []
            for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", source):
                command, environment = match.groups()
                if command == "begin":
                    stack.append(environment)
                else:
                    self.assertTrue(stack, msg=f"Unexpected end of {environment} in {name}")
                    self.assertEqual(
                        stack.pop(), environment, msg=f"Mismatched environment in {name}"
                    )
            self.assertEqual(stack, [], msg=f"Unclosed environments in {name}")

    def test_revtex_longtable_uses_compatible_array_parser(self):
        self.assertIn(r"\usepackage{array}[=2016-10-06]", self.source)
        self.assertNotIn(r"\usepackage{booktabs,tabularx", self.source)

    def test_srq_and_ternary_rate_contracts_are_explicit(self):
        self.assertIn(r"P_S(a,b,c|x,y,z)", self.source)
        self.assertIn(r"P_L(a,b,c|x,y,z)", self.source)
        self.assertIn(r"p_B(b|y,\omega)p_C(c|z,\omega)", self.source)
        self.assertIn(r"\EC&=1-L(\nu)\min\{\eta_B,\eta_C\}", self.main)
        self.assertNotIn(r"\label{cor:hybrid-ideal}", self.source)
        self.assertIn(r"\label{eq:ideal-rate}", self.main)
        self.assertIn(r"K_{\rPhase,\mathrm{ideal}}", self.main)
        self.assertIn(r"\eta_B\eta_C+\min", self.source)
        self.assertIn(r"(\sqrt5-1)/2", self.main)

    def labeled_environment(self, source, environment, label):
        blocks = [
            block for block in re.findall(
                rf"\\begin\{{{environment}\*?\}}.*?\\end\{{{environment}\*?\}}",
                source, flags=re.DOTALL,
            ) if rf"\label{{{label}}}" in block
        ]
        self.assertEqual(len(blocks), 1, msg=f"Expected one {environment}: {label}")
        return blocks[0]

    def test_protocol_family_table_precedes_protocol_box_in_section_two(self):
        framework = self.main.split(
            r"\section{Tripartite Routed Conference Key Agreement}", 1
        )[1].split(r"\section{", 1)[0]
        family = self.labeled_environment(framework, "table", "tab:protocol-family")
        protocol = self.labeled_environment(framework, "protocolbox", "prot:routed")
        self.assertLess(framework.index(family), framework.index(protocol))
        for variant in (r"\rPCHSH", r"\rPhase", r"\rHybrid"):
            self.assertIn(variant, family)

    def test_first_input_description_gives_honest_physical_meanings(self):
        first_inputs = self.main.split("Alice has measurements", 1)[1].split(
            r"\begin{figure}", 1
        )[0]
        for expression in ("honest GHZ", r"ZZZ", r"XXX", r"(0,2,1)", r"(1,3,0)"):
            self.assertIn(expression, first_inputs)
        self.assertIn("uncharacterized", first_inputs)
        self.assertIn(r"A_0", first_inputs)
        self.assertIn(r"B_2^L", first_inputs)
        self.assertIn(r"C_1^L", first_inputs)

    def test_angle_policy_explains_honest_target_retuning(self):
        angle_policy = self.main.split(r"\label{eq:angle-policy}", 1)[1].split(
            r"\subsection{", 1
        )[0]
        for fragment in ("retunes", "not a global optimization", "remain fixed"):
            self.assertIn(fragment, angle_policy)
        self.assertIn(r"\rPCHSH", angle_policy)
        self.assertIn(r"\rHybrid", angle_policy)
        self.assertIn(r"\rPhase", angle_policy)

    def test_formal_srq_model_and_obstruction_are_unique_main_statements(self):
        framework = self.main.split(
            r"\section{Tripartite Routed Conference Key Agreement}", 1
        )[1].split(r"\section{", 1)[0]
        definition = self.labeled_environment(framework, "definition", "def:srq")
        obstruction = self.labeled_environment(
            framework, "proposition", "prop:srq-obstruction"
        )
        for label in ("def:srq", "eq:srq-short", "eq:srq-long", "prop:srq-obstruction"):
            with self.subTest(label=label):
                self.assertEqual(self.source.count(rf"\label{{{label}}}"), 1)
                self.assertNotIn(rf"\label{{{label}}}", self.appendix)
        for expression in (
            r"P_S(a,b,c|x,y,z)", r"P_L(a,b,c|x,y,z)",
            r"p_B(b|y,\omega)p_C(c|z,\omega)", r"\{N_\omega\}",
            r"\perpout",
        ):
            self.assertIn(expression, definition)
        self.assertIn("route-blind", definition)
        obstruction = " ".join(obstruction.split())
        self.assertRegex(obstruction, r"(?i)fixed.*(?:key inputs|raw)")
        self.assertRegex(obstruction, r"(?i)(?:only|alone).*Alice|Alice.*(?:only|alone)")
        self.assertIn("preprocessing", obstruction)

    def test_srq_obstruction_proof_remains_in_appendix(self):
        srq = self.appendix.split(r"\label{app:srq}", 1)[1].split(r"\section{", 1)[0]
        self.assertIn(r"\label{lem:si-simulation}", srq)
        self.assertIn(r"\label{eq:si-degrading}", srq)
        self.assertIn(r"\label{eq:si-simulation-transcript}", srq)
        proofs = re.findall(r"\\begin\{proof\}.*?\\end\{proof\}", srq, re.DOTALL)
        self.assertTrue(any(
            r"\eqref{eq:srq-long}" in proof and r"\ref{lem:si-simulation}" in proof
            for proof in proofs
        ), msg="SRQ obstruction proof must remain in the SRQ appendix")
        self.assertIn("disjoint rounds", srq)
        self.assertIn("private randomness", srq)
        self.assertIn("Receiver feedback", srq)
        self.assertIn("retained-record-dependent selection", srq)

    def test_honest_entropies_and_ideal_statements_remain_in_main(self):
        equations = re.sub(r"\s+", "", self.main.replace("&", ""))
        for expression in (
            r"\EC=\max\{H(A_0|B_2),H(A_0|C_1)\}",
            r"K=H(A_0|E)-\EC",
            r"H(A_1|B_3,C_0)=1-\eta_B\eta_C L(\nu)",
            r"H(A_0|B_2)=1-\eta_B L(\nu)",
            r"H(A_0|C_1)=1-\eta_C L(\nu)",
            r"\EC=1-L(\nu)\min\{\eta_B,\eta_C\}",
            r"H(A_0|E)\geq\eta_B\eta_C",
            r"K_{\rPhase,\mathrm{ideal}}=K_{\rHybrid,\mathrm{ideal}}=\eta_B\eta_C+\min\{\eta_B,\eta_C\}-1",
        ):
            self.assertIn(re.sub(r"\s+", "", expression), equations)
        theorem = self.labeled_environment(self.main, "theorem", "thm:ideal-phase")
        for variant in (r"\rPhase", r"\rHybrid"):
            self.assertIn(variant, theorem)
        self.assertNotIn(r"\begin{corollary}", self.main)
        self.assertIn(r"\nu=\eta_S=1", theorem)
        self.assertIn(r"\eta_A=1", theorem)
        self.assertIn("fixed raw data and communication class", theorem)
        for label in ("eq:analytic-entropy", "eq:phase-cost", "eq:ternary-reconciliation",
                      "eq:target-ec", "eq:ideal-entropy", "eq:ideal-rate"):
            self.assertIn(rf"\label{{{label}}}", self.main)

    def test_ideal_matching_attack_and_one_way_converse_are_preserved(self):
        ideal = self.appendix.split(r"\label{app:ideal}", 1)[1].split(
            r"\label{app:phase-local}", 1
        )[0]
        for label in (
            "eq:si-ideal-lower", "eq:si-erasure-isometry",
            "eq:si-converse-start", "eq:si-erasure-converse", "eq:si-ideal-rate",
        ):
            self.assertIn(rf"\label{{{label}}}", ideal)
        self.assertIn(r"\eta_B\eta_C", ideal)
        self.assertRegex(ideal, r"(?i)preprocess")
        self.assertRegex(ideal, r"(?i)telescop")
        self.assertRegex(ideal, r"(?i)disjoint[^.]*test")
        self.assertRegex(ideal, r"(?i)attack[^.]*reproduces all ideal remote")

    def test_bff_optimization_and_unconditional_polynomial_are_explicit(self):
        bff = self.appendix.split(r"\label{app:bff}", 1)[1].split(r"\section{", 1)[0]
        optimization = self.labeled_environment(bff, "equation", "eq:si-bff-optimization")
        self.assertRegex(optimization, r"\\(?:inf|min)")
        constraints = re.sub(r"\s+", "", optimization)
        self.assertIn(r"\langlew\rangle=p_w", constraints)
        self.assertIn(r"O=O^\dagger=O^2", constraints)
        polynomial = self.labeled_environment(bff, "equation", "eq:si-bff-polynomial")
        equations = re.sub(r"\s+", "", polynomial)
        for expression in (
            r"Z_{a,i}+Z_{a,i}^\dagger+(1-t_i)Z_{a,i}^\daggerZ_{a,i}",
            r"+t_iZ_{a,i}Z_{a,i}^\dagger",
        ):
            self.assertIn(expression, equations)
        self.assertRegex(bff, r"(?i)unconditional")
        self.assertIn(r"\label{eq:si-bff-bridge}", bff)
        self.assertIn(r"\|Z_{a,i}\|\leq\alpha_i", re.sub(r"\s+", "", bff))
        self.assertIn(r"1/(1-t_i)", bff)

    def test_bff_algebra_enrichment_and_parameter_settings_are_preserved(self):
        bff = self.appendix.split(r"\label{app:bff}", 1)[1].split(r"\section{", 1)[0]
        prose = " ".join(bff.split())
        equations = re.sub(r"\s+", "", bff)
        self.assertIn(r"[A_x^+,O]=[A_x^+,Z_{a,i}]=[O_L,Z_{a,i}]=0", equations)
        for party in ("remote", "auxiliar"):
            self.assertRegex(
                prose,
                rf"(?i)no commutation[^.]*nearby[^.]*{party}"
                rf"|nearby[^.]*not assumed to commute[^.]*{party}",
            )
        self.assertRegex(prose, r"(?i)degree[- ]two|level[- ]two|level\s*\(?2")
        self.assertRegex(prose, r"(?i)enrich")
        for word in (r"A_x\^\+O(?:_L)?Z_a", r"A_x\^\+O(?:_L)?Z_a\^\\dagger"):
            self.assertRegex(bff, word)
        for polynomial in (
            r"\alpha_i^2\id-Z_{a,i}^\daggerZ_{a,i}",
            r"\alpha_i^2\id-Z_{a,i}Z_{a,i}^\dagger",
        ):
            self.assertIn(polynomial, equations)
        for matrix in (r"\Gamma", r"\Gamma(g_{a,1})", r"\Gamma(g_{a,2})"):
            self.assertIn(matrix + r"\succeq0", equations)
        for setting in (r"m=4", r"m=2", r"\nu=\eta_S=0.99", r"\eta_B=\eta_C=0.78"):
            self.assertIn(setting, equations)
        self.assertIn("(17,4,8)", equations)
        self.assertRegex(
            prose,
            r"m=2.{0,80}(?:bound.{0,30}(?<![\d.])5(?![\d.])"
            r"|\\alpha(?:_i|_1)?=5(?![\d.]))",
        )
        self.assertRegex(prose, r"(?i)control")
        self.assertIn("MOSEK", bff)
        self.assertIn(r"10^{-7}", bff)
        self.assertRegex(prose, r"(?i)finite[- ]order[^.]*not[^.]*converg")

    def test_protocol_causal_order_and_records_are_explicit(self):
        self.assertIn("causally unavailable to the source and Alice's device", self.main)
        self.assertNotIn("A public random branch variable", self.main)
        self.assertIn("The selected settings are supplied before the devices produce outputs", self.main)
        self.assertIn("Alice's binary raw string", self.main)
        self.assertIn("ternary records as separate decoding information", self.main)
        for premise in (
            "asymptotic i.i.d.", "collective attacks", "authenticated public channel",
            "Bob and Charlie do not provide feedback",
            "only Alice sends public messages",
        ):
            self.assertIn(premise, " ".join(self.main.split()))

    def test_numerical_measurement_labels_are_consistent(self):
        generated = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (PAPER / "generated").glob("*.tex")
        )
        for source in (self.main, self.appendix, generated):
            self.assertNotRegex(source, r"[AB]_[XZ]")
            self.assertNotIn("Alice-forward", source)
            self.assertNotIn("one-way erasure-channel", source)
        self.assertIn(r"A_0&=Z", self.main)
        self.assertIn(r"B_2^L&=Z", self.main)
        self.assertIn(r"B_3^L=X", self.main)
        self.assertIn(r"C_1^L=Z", self.main)
        self.assertIn(r"\phi_L&=\langle A_1B_3^{L}C_0^{L}\rangle", self.main)

    def test_reference_point_is_not_claimed_to_violate_the_scalar_srq_family(self):
        self.assertNotIn(r"\label{eq:no-scalar-srq-violation}", self.main)
        self.assertIn(r"\label{eq:no-scalar-srq-violation}", self.appendix)
        self.assertIn(r"\overline W_L^{(\lambda)}", self.appendix)

    def test_scalar_srq_family_and_explicit_phi_formula_are_appendix_only(self):
        for expression in (r"\overline W_L^{(\lambda)}", r"\Phi_\lambda"):
            self.assertNotIn(expression, self.main)
        for label in ("eq:si-lambda-definitions", "eq:si-lambda-bound", "thm:si-lambda"):
            self.assertEqual(self.appendix.count(rf"\label{{{label}}}"), 1)
        equations = re.sub(r"\s+", "", self.appendix)
        for expression in (
            r"\overline W_L^{(\lambda)}=\lambda P_{0,L}+P_{1,L}",
            r"\overline W_L^{(\lambda)}\leq\Phi_\lambda(\beta_S)",
            r"\Phi_\lambda(\beta_S)=\sqrt{\lambda^2+(1-\lambda^2)",
            r"\frac{\beta_S+\sqrt{2-\beta_S^2}}2",
        ):
            self.assertIn(re.sub(r"\s+", "", expression), equations)

    def test_full_remote_bell_witness_is_not_confused_with_the_scalar_family(self):
        witness = self.appendix.split(r"\label{app:remote-bell-witness}", 1)[1]
        witness = re.split(r"\\(?:sub)?section\{", witness, maxsplit=1)[0]
        equations = re.sub(r"\s+", "", witness)
        self.assertIn(r"S_{C+}\leq2P(C_0=+)", equations)
        self.assertRegex(witness, r"1\.0838\d*")
        self.assertIn(r"\nu_0\eta_0", equations)
        self.assertTrue(
            r"\cos\theta_0+\sin\theta_0" in equations
            or (r"1+\eta_0" in equations and r"\sqrt{1+\eta_0^2}" in equations),
            msg="Keep the physical target evaluation of the lifted Bell witness",
        )
        prose = " ".join(witness.split())
        self.assertRegex(prose, r"(?i)unnormalized")
        self.assertRegex(prose, r"(?i)without[^.]*fair[- ]sampling|no fair[- ]sampling")
        self.assertRegex(prose, r"(?i)deterministic local|local deterministic")

    def test_calibration_counterexample_does_not_claim_to_match_the_target(self):
        calibration = self.main.split(r"\label{eq:near-score}", 1)[1].split(
            r"\label{eq:analytic-entropy}", 1
        )[0]
        self.assertIn("Bell pair", calibration)
        self.assertIn("Charlie", calibration)
        self.assertIn("always outputs", calibration)
        self.assertIn(r"H(A_0|C_1)=1", calibration)
        self.assertIn("does not reproduce the GHZ key data", calibration)

    def test_score_calibration_and_joint_statistics_are_compared_at_same_target(self):
        comparison = self.main.split(r"\label{sec:routing-secrecy}", 1)[1].split(
            r"\subsection{", 1
        )[0]
        self.assertIn(r"\eqref{eq:analytic-entropy}", comparison)
        self.assertIn("0.06316", comparison)
        self.assertIn("same target", comparison)
        self.assertIn("joint", comparison)
        self.assertIn("additional constraints", comparison)
        self.assertIn(r"H(A_0|E)\geq0.3327199", comparison)
        self.assertNotIn("all remote probabilities in the BFF", comparison)
        prose = " ".join(comparison.split())
        self.assertRegex(prose, r"(?i)score[- ](?:based|only)")
        self.assertRegex(prose, r"(?i)below[^.]*reconciliation")
        self.assertRegex(prose, r"(?i)unchanged reconciliation|same[^.]*key records")

    def test_ideal_mwrf_mechanism_keeps_matching_bounds_and_raw_records(self):
        proof = self.appendix.partition(r"\label{app:ideal-mwrf}")[2]
        self.assertTrue(proof, msg="Missing ideal M/W/R/F proof")
        proof = re.split(r"\\(?:sub)?section\{", proof, maxsplit=1)[0]
        self.assertIn(r"\eta_B\eta_C", proof)
        self.assertIn(r"H(A_1|B_3,C_0)", proof)
        self.assertIn("coherent-erasure", proof)
        self.assertIn("dephasing", proof)
        self.assertIn(r"\label{eq:si-ideal-phase-saturation}", proof)
        self.assertIn(r"\label{eq:si-ideal-mwrf-rate}", proof)
        self.assertRegex(proof, r"(?i)attain|match")
        self.assertRegex(proof, r"(?i)not a minimality|minimality.*not")
        equations = re.sub(r"\s+", "", proof)
        self.assertIn(r"H(A_0|E)\geq\tau", equations)
        self.assertIn(r"K_{\mathrmM}=K_{\mathrmR}=0", equations)

    def test_routing_advantage_has_an_analytical_example(self):
        start = self.main.index(r"\section{Key Rates and Routing Advantage}")
        end = self.main.index(r"\section{Discussion}")
        results = self.main[start:end]
        self.assertLess(results.index(r"\label{sec:routing-secrecy}"),
                        results.index(r"\label{sec:mwrf}"))
        self.assertLess(results.index(r"\label{sec:mwrf}"),
                        results.index("Performance of the protocol family"))
        self.assertIn(r"A_0=B_2=C_1=k", results)
        self.assertIn(r"C_0=st", results)
        self.assertIn(r"P_L=(1-\gamma)Q_{\mathrm{GHZ}}+\gamma P_{\mathrm{loc}}", results)
        self.assertNotIn("288", results)
        self.assertNotIn("123", results)

    def test_noisy_activation_is_a_scoped_main_proposition(self):
        results = self.main.split(r"\label{sec:routing-secrecy}", 1)[1].split(
            r"\subsection{", 1
        )[0]
        proposition = self.labeled_environment(results, "proposition", "thm:noisy-activation")
        self.assertEqual(self.source.count(r"\label{thm:noisy-activation}"), 1)
        proposition = " ".join(proposition.split())
        self.assertRegex(proposition, r"(?i)fixed.*raw")
        self.assertRegex(proposition, r"(?i)(?:only|alone).*Alice|Alice.*(?:only|alone)")
        self.assertIn("preprocessing", proposition)
        self.assertIn(r"\nu=\eta_S=0.99", proposition)
        self.assertIn(r"\eta_B=\eta_C=0.78", proposition)
        equations = re.sub(r"\s+", "", results.replace("&", ""))
        for expression in (r"H(A_0|E)\geq0.3327199", r"\EC\leq0.2554235"):
            self.assertIn(expression, equations)
        self.assertIn(r"\label{eq:strict-separation}", proposition)
        self.assertIn(r"K_{\mathrm{remote}}=0", proposition)
        self.assertIn(r"K_{\mathrm{routed}}>0.07729", proposition)
        self.assertIn(r"\label{eq:si-reference-rate}", self.appendix)

    def test_full_remote_attack_keeps_the_physical_mixture_and_channel_simulation(self):
        attack = self.appendix.split(r"\label{app:attack}", 1)[1].split(
            r"\label{app:remote-bell-witness}", 1
        )[0]
        mixture = self.labeled_environment(attack, "equation", "eq:si-exact-attack")
        equations = re.sub(r"\s+", "", attack)
        mixture_equations = re.sub(r"\s+", "", mixture)
        self.assertRegex(mixture, r"q_?0?Q_\{\\mathrm\{GHZ\}\}")
        self.assertRegex(
            mixture_equations,
            r"\\sum.*w_(\{[^}]+\}|\\[A-Za-z]+|[A-Za-z])D_\1",
        )
        self.assertRegex(mixture_equations, r"q(?:_0)?,w_.*\\geq0")
        self.assertRegex(mixture_equations, r"q(?:_0)?\+\\sum.*w_.*=1")
        prose = " ".join(attack.split())
        for pattern in (
            r"\bnonnegative\b|\bnonnegativity\b|\bpositivity\b",
            r"normaliz|sum.*=\s*1", r"\bbalanced\b", r"deterministic",
            r"complete[^.]*table|all[^.]*probabilit|every conditional probability",
            r"flip", r"preprocess", r"(?:only|alone)[^.]*Alice|Alice[^.]*(?:only|alone)",
        ):
            self.assertRegex(prose, "(?i)" + pattern)
        self.assertRegex(prose, r"0\.16(?:2|199\d*)")
        self.assertRegex(prose, r"0\.8380\d*")
        self.assertRegex(equations, r"\\eta_0/\\gamma(?:_0)?")
        self.assertRegex(equations, r"\\epsilon=\(1-\\nu_0\)/2(?:=1/200)?")
        self.assertIn(r"\ref{lem:si-simulation}", attack)

    def test_main_mwrf_table_is_a_four_row_physical_comparison(self):
        table = self.labeled_environment(self.main, "table", "tab:mwrf")
        tabular = re.search(
            r"\\begin\{tabular\}[^\n]*\n(.*?)\\end\{tabular\}", table, re.DOTALL
        )
        self.assertIsNotNone(tabular, msg="Missing physical M/W/R/F tabular")
        rows = [row for row in tabular[1].split(r"\\") if "&" in row]
        self.assertEqual(len(rows), 5, msg="Expected a header and exactly M/W/R/F rows")
        header, *data = rows
        self.assertRegex(header, r"(?i)data|observations")
        self.assertRegex(header, r"(?i)dephas")
        self.assertRegex(header, r"(?i)ideal")
        self.assertRegex(header, r"(?i)entropy|H\(A_0\|E\)")
        columns = len(header.split("&"))
        self.assertIn(columns, (3, 4), msg="Data, optional retained information, dephasing, entropy")
        for letter, row in zip("MWRF", data):
            with self.subTest(data=letter):
                cells = row.split("&")
                self.assertEqual(len(cells), columns, msg="M/W/R/F rows must match the header")
                self.assertRegex(cells[0], rf"\b{letter}\b")
                compatibility = set(re.findall(r"[a-z]+", cells[-2].lower()))
                expected = {"yes", "compatible"} if letter in "MR" else {
                    "no", "excluded", "incompatible",
                }
                self.assertTrue(compatibility & expected, msg=f"Wrong dephasing status: {letter}")
                if letter in "MR":
                    self.assertRegex(cells[-1], r"(?<![\d.])0(?![\d.])")
                else:
                    self.assertIn(r"\eta_B\eta_C", re.sub(r"\s+", "", cells[-1]))

    def test_mwrf_triples_stay_main_but_coordinate_inversion_is_appendix_only(self):
        self.assertIn(r"T_{x,y,z}", self.main)
        self.assertIn(r"\label{eq:mwrf}", self.main)
        equations = re.sub(r"\s+", "", self.main)
        for expression in (
            r"W=M\cup\{T_{1,0,0},T_{1,1,0},T_{1,3,0}\}",
            r"R=F\setminus\{T_{1,0,0},T_{1,1,0},T_{1,3,0}\}",
        ):
            self.assertIn(expression, equations)
        appendix_equations = re.sub(r"\s+", "", self.appendix)
        for pattern in (r"V(?:\(o\))?=\(1,-1,0\)", r"D(?:\(o\))?=\(1,1,0\)"):
            with self.subTest(coordinate=pattern):
                self.assertNotRegex(equations, pattern)
                self.assertRegex(appendix_equations, pattern)
        inversions = ("P_+=(D+V)/2", "P_-=(D-V)/2", r"P_\pm=(D\pmV)/2")
        for expression in inversions + (r"P_\perp=I-D",):
            self.assertNotIn(expression, equations)
        self.assertTrue(
            all(expression in appendix_equations for expression in inversions[:2])
            or inversions[2] in appendix_equations,
            msg="Appendix must retain the ternary coordinate-effect inversion",
        )
        self.assertIn(r"P_\perp=I-D", appendix_equations)
        self.assertNotIn(r"\label{eq:coherent-differences}", self.main)
        self.assertIn(r"\label{eq:coherent-differences}", self.appendix)
        for expression in (
            r"\Delta T_{1,0,0}=\nu\eta_B\eta_C\sin\theta_L",
            r"\Delta T_{1,1,0}=-\nu\eta_B\eta_C\sin\theta_L",
            r"\Delta T_{1,3,0}=\nu\eta_B\eta_C",
        ):
            self.assertIn(re.sub(r"\s+", "", expression), appendix_equations)

    def test_removed_numerical_tables_are_not_expanded_into_the_manuscript(self):
        body = self.source.partition(r"\begin{document}")[2]
        for command in (r"\representativepointrows", r"\radausensitivityrows"):
            self.assertNotIn(command, body)
        for filename in (
            "representative_point_rows.tex", "radau_sensitivity_rows.tex",
            "attack_weights_table.tex", "proof_manifest_table.tex",
            "m4_affine_table.tex", "archived_m2_affine_table.tex",
        ):
            self.assertNotIn(filename, body)

    def test_information_comparison_precedes_standalone_discussion(self):
        results_start = self.main.index(r"\section{Key Rates and Routing Advantage}")
        discussion_start = self.main.index(r"\section{Discussion}")
        backmatter_start = self.main.index(r"\section*{Code availability}")
        results = self.main[results_start:discussion_start]
        discussion = self.main[discussion_start:backmatter_start]
        for label in ("sec:mwrf", "eq:mwrf", "tab:mwrf"):
            self.assertIn(rf"\label{{{label}}}", results)
            self.assertNotIn(rf"\label{{{label}}}", discussion)
        self.assertNotIn(r"\subsection", discussion)

    def test_no_internal_experiment_ids_leak_into_main_text(self):
        self.assertNotRegex(self.main, r"\bat S[1234]\b")
        self.assertNotIn("representative point S", self.main)

    def test_continuous_parameter_box_is_not_a_manuscript_claim(self):
        self.assertNotIn(r"\label{thm:si-continuous}", self.compiled)
        self.assertNotIn(r"\label{app:continuous}", self.compiled)
        self.assertNotIn("Continuous noisy separation", self.compiled)
        self.assertNotIn("0.999", self.compiled)
        self.assertNotIn("0.01619295", self.compiled)

    def test_archived_affine_coefficients_remain_intact_without_box_claims(self):
        affine = json.loads(
            (
                ROOT
                / "source_data"
                / "certificates"
                / "archived_m2"
                / "near_m2.affine.json"
            ).read_text(encoding="utf-8")
        )
        coefficient = {
            tuple(item["word"]): Fraction(item["value"])
            for item in affine["coefficients"]
        }
        grouped = {
            "C0": coefficient[()] + (coefficient[(0,)] + coefficient[(1,)]) / 2,
            "Cm": coefficient[(2,)]
            + coefficient[(3,)]
            + (
                coefficient[(0, 2)]
                + coefficient[(0, 3)]
                + coefficient[(1, 2)]
                + coefficient[(1, 3)]
            )
            / 2,
            "CZ": (coefficient[(0, 2)] + coefficient[(0, 3)]) / 4,
            "CX": (coefficient[(1, 2)] - coefficient[(1, 3)]) / 4,
            "Ctau": (coefficient[(4,)] + coefficient[(5,)]) / 2
            + (coefficient[(0, 4)] + coefficient[(0, 5)]) / 4
            + (coefficient[(1, 4)] + coefficient[(1, 5)]) / 4,
            "Cphi": (coefficient[(1, 4)] - coefficient[(1, 5)]) / 4,
            "delta": Fraction(affine["residual_penalty"]),
        }
        expected = {
            "C0": Fraction(-427156159036595387, 144115188075855872),
            "Cm": Fraction(19667907614480465, 72057594037927936),
            "CZ": Fraction(375601792294839715, 288230376151711744),
            "CX": Fraction(369188273980106757, 288230376151711744),
            "Ctau": Fraction(-27074923774278813, 18014398509481984),
            "Cphi": Fraction(500622691152273507, 288230376151711744),
            "delta": Fraction(429397180060949, 216172782113783808),
        }
        self.assertEqual(grouped, expected)

    def test_generated_affine_tables_are_complete(self):
        for stem, prefix, nodes in (
            ("m4_affine_table", "m4", range(3)),
            ("archived_m2_affine_table", "archived_m2", (None,)),
        ):
            table = (PAPER / "generated" / (stem + ".tex")).read_text(encoding="utf-8")
            for node in nodes:
                filename = "near_m2.affine.json" if node is None else f"node_{node}.affine.json"
                affine = json.loads((ROOT / "source_data" / "certificates" / prefix / filename)
                                    .read_text(encoding="utf-8"))
                marker = "ref." if node is None else str(node + 1)
                rows = re.findall(rf"^{re.escape(marker)} &", table, flags=re.MULTILINE)
                self.assertEqual(len(rows), len(affine["coefficients"]) + 1)

    def test_generated_attack_table_is_complete(self):
        table = (PAPER / "generated" / "attack_weights_table.tex").read_text(
            encoding="utf-8"
        )
        rows = re.findall(r"^(\d+) &", table, flags=re.MULTILINE)
        self.assertEqual(rows, [str(index) for index in range(1, 124)])

        attack = json.loads(
            (ROOT / "source_data" / "attack" / "attack_eta078.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(attack["local_strategies"]), 123)
        self.assertEqual(len(attack["local_weights"]), 123)
        self.assertEqual(attack["verification"]["probabilities"], 288)

    def test_frozen_proof_paths_are_local(self):
        config = json.loads((ROOT / "config" / "paper_suite.json").read_text())
        for relative in config["external_archives"].values():
            self.assertTrue(relative.startswith("source_data/"))
            self.assertTrue((ROOT / relative).is_file())

    def test_no_known_latex_escape_damage(self):
        combined = self.source
        self.assertNotIn(",qquad", combined)
        self.assertNotIn(r"\eta_B,eta_C", combined)
        self.assertNotIn("Assigning the signed values Assigning the signed values", combined)

    def test_main_text_uses_only_the_final_curve_figures(self):
        stems = (
            "routed_vs_long_only_rpchsh",
            "routed_vs_long_only_rphase",
            "routed_vs_long_only_rhybrid",
            "routed_protocols_near_efficiency",
            "routed_protocols_visibility",
        )
        for stem in stems:
            self.assertIn(stem + ".png", self.main)
        self.assertNotIn("ideal_rate_mechanism", self.source)
        self.assertNotIn(r"\label{fig:ideal-rate}", self.source)
        self.assertNotIn("protocol_rates.pdf", self.main)
        self.assertNotIn("strict_separation.pdf", self.main)
        self.assertIn("0.077", self.main)
        self.assertIn(r"\label{tab:mwrf}", self.main)

    def test_reference_rate_keeps_conservative_entropy_and_cost_bounds(self):
        reference = self.labeled_environment(
            self.appendix, "equation", "eq:si-reference-rate"
        )
        equations = re.sub(r"\s+", "", reference)
        self.assertRegex(equations, r"H\(A_0\|E\)\\geq0\.3327199\d*")
        self.assertRegex(equations, r"\\EC\\leq0\.255423(?:5|4601\d*)")
        self.assertRegex(equations, r"K(?:>|\\geq)0\.07729\d*")

    def test_srq_illustration_uses_the_nearby_score_scan(self):
        figures = re.findall(
            r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", self.appendix
        )
        self.assertEqual(figures, ["tripartite_parity_tradeoff_vs_betaS.png"])
        for suffix in (".pdf", ".png"):
            path = PAPER / "figures" / ("tripartite_parity_tradeoff_vs_betaS" + suffix)
            self.assertTrue(path.is_file(), msg=f"Missing SRQ figure asset: {path}")
        self.assertIn(r"\label{fig:si-srq-tradeoff}", self.appendix)
        self.assertIn(r"\label{eq:si-conditional-quantum}", self.appendix)
        self.assertNotIn(r"\ref{fig:si-srq-maximal}", self.appendix)

    def test_all_included_external_figures_are_existing_pngs(self):
        figures = re.findall(
            r"\\(?:includegraphics(?:\[[^]]*\])?|resultfigure)\{([^}]+)\}",
            self.source,
        )
        figures = [name for name in figures if not name.startswith("#")]
        self.assertEqual(len(figures), 6)
        for name in figures:
            self.assertEqual(Path(name).suffix, ".png", msg=name)
            self.assertTrue((PAPER / "figures" / name).is_file(), msg=name)


if __name__ == "__main__":
    unittest.main()
