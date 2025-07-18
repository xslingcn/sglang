import json
from pathlib import Path
from typing import Optional


class KernelCallLogger:
    def __init__(
        self,
        definition: str,
        type: str,
        output_dir: str = "/home/xsling/Code/flashinfer-bench/dataset/traces/workloads",
        environment: Optional[dict] = None,
    ):
        self.definition = definition
        self.type = type
        self.env = environment or {}
        self.output_dir = Path(output_dir) / type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.call_index = 0
        self.log_file = self.output_dir / f"{self.definition}.workload.jsonl"

    def log_call(self, inputs: dict, axes: dict):
        print(f"Logging call {self.call_index} for {self.definition}")
        input_types = {
            name: {
                # for gemm and grouped gemm, input values are randomly generated given shape
                "type": "random" 
            }
            for name, tensor in inputs.items()
        }

        entry = {
            "axes": axes,
            "inputs": input_types
        }

        trace = {
            "definition": self.definition,
            "solution": "",
            "workload": entry,
            "evaluation": {}
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(trace) + "\n")

        self.call_index += 1