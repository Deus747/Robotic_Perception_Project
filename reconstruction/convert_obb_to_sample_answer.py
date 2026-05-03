import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "obb_reconstruction_inputs" / "yolo26_9class_nonml_geometric_refit" / "obb_estimates.json"
TARGET = ROOT / "obb_reconstruction_inputs" / "yolo26_9class_nonml_geometric_refit" / "sample_answers.json"

ENTITY_NAME_MAP = {
    "vga_port": "vga_socket",
    "ethernet_port": "ethernet_socket",
    "power_socket": "power_socket",
    "bottle": "bottle",
    "key": "key",
    "glass": "glass",
    "ps": "ps",
    "sticker": "sticker",
}


def main():
    estimates = json.loads(SOURCE.read_text())
    answers = []

    for source_name, obb in estimates.items():
        entity_name = ENTITY_NAME_MAP.get(source_name, source_name)
        answers.append(
            {
                "entity": entity_name,
                "obb": {
                    "center": obb["center"],
                    "extent": obb["extent"],
                    "rotation": obb["rotation"],
                },
            }
        )

    TARGET.write_text(json.dumps(answers, indent=2))
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
