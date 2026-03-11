import os
import pathlib
import torch

from pysdd.sdd import SddManager, Vtree


from evaluator import ProbSemiring, LogProbSemiring, evaluate_formula, EPSILON


def test(outdir):

    vtree = Vtree(var_count=4, vtree_type="balanced")
    manager = SddManager.from_vtree(vtree)

    a, b, c, d = manager.vars

    # Build SDD for formula
    formula = (a & b) | (b & c) | (c & d)

    # Visualize SDD and Vtree
    with open(outdir / "sdd.dot", "w+") as out:
        print(formula.dot(), file=out)
    with open(outdir / "vtree.dot", "w+") as out:
        print(vtree.dot(), file=out)

    # evaluate SDD
    values = torch.rand(formula.manager.var_count())
    prob_semiring = ProbSemiring()
    prob_result = evaluate_formula(formula, values, prob_semiring)
    print(prob_result)

    logprob_semiring = LogProbSemiring()
    # logprob_result = evaluate_formula(formula, values, logprob_semiring)
    # print(logprob_result)

    # print(torch.isclose(torch.log(prob_result + EPSILON), logprob_result))


if __name__ == "__main__":
    current_file = pathlib.Path(__file__)
    current_dir = current_file.parent

    outdir = current_dir / "outdir"
    os.makedirs(outdir, exist_ok=True)
    test(outdir)
