"""Where Core ML puts each op of a compiled model (MLComputePlan): a count
of op types on each device, and the ops kept off the Neural Engine.

  placement.py MODEL.mlmodelc [--units ne|all]
"""

import argparse
import collections

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="a compiled .mlmodelc (xcrun coremlcompiler compile)")
    ap.add_argument("--units", default="ne", choices=["ne", "all"])
    ap.add_argument("--function", default=None)
    args = ap.parse_args()
    units = ct.ComputeUnit.CPU_AND_NE if args.units == "ne" else ct.ComputeUnit.ALL
    plan = MLComputePlan.load_from_path(args.model, compute_units=units)
    prog = plan.model_structure.program
    fns = prog.functions
    name = args.function or next(iter(fns))
    counts = collections.Counter()
    off = collections.Counter()
    cost = collections.Counter()
    for op in fns[name].block.operations:
        if op.operator_name in ("const",):
            continue
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = (
            type(u.preferred_compute_device)
            .__name__.replace("ML", "")
            .replace("ComputeDevice", "")
            if u
            else "?"
        )
        c = plan.get_estimated_cost_for_mlprogram_operation(op)
        counts[(op.operator_name, dev)] += 1
        cost[dev] += c.weight if c else 0
        if dev != "NeuralEngine":
            off[op.operator_name] += 1
    print("function %s: %d ops" % (name, sum(counts.values())))
    for (op, dev), n in sorted(counts.items()):
        print("  %-16s %-14s %d" % (op, dev, n))
    print(
        "estimated cost by device: %s"
        % ", ".join("%s %.3f" % kv for kv in cost.items())
    )
    print("off the Neural Engine: %s" % (dict(off) or "none"))


if __name__ == "__main__":
    main()
