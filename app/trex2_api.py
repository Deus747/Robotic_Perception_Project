import base64
import copy
import time
from io import BytesIO
from typing import Dict, List, Union

import requests
from PIL import Image


def encode_image(image):
    if isinstance(image, str):
        if image.startswith("data:image/") and ";base64," in image:
            return image.split(";base64,", 1)[1]
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    if isinstance(image, Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    raise TypeError("Input must be a file path (str) or PIL.Image.Image object.")


class TRex2APIWrapper:
    def __init__(self, token: str):
        self.headers = {"Content-Type": "application/json", "Token": token}

    def call_api(self, task_dict):
        resp = requests.post(
            url="https://api.deepdataspace.com/v2/task/trex/detection",
            json=task_dict,
            headers=self.headers,
        )
        try:
            json_resp = resp.json()
        except Exception as e:
            snippet = resp.text[:500] if hasattr(resp, "text") else "<no text>"
            raise RuntimeError(
                f"T-Rex2 API returned a non-JSON response. status={resp.status_code} error={e} body_snippet={snippet}"
            ) from e
        if json_resp.get("msg") != "ok":
            raise RuntimeError(f"API call failed with error: {json_resp}")
        task_uuid = json_resp["data"]["task_uuid"]

        while True:
            resp = requests.get(
                f"https://api.deepdataspace.com/v2/task_status/{task_uuid}",
                headers=self.headers,
            )
            try:
                json_resp = resp.json()
            except Exception as e:
                snippet = resp.text[:500] if hasattr(resp, "text") else "<no text>"
                raise RuntimeError(
                    f"T-Rex2 task-status API returned a non-JSON response. status={resp.status_code} error={e} body_snippet={snippet}"
                ) from e
            if json_resp["data"]["status"] not in ["waiting", "running"]:
                break
            time.sleep(1)

        if json_resp["data"]["status"] == "failed":
            raise RuntimeError(f"API call failed with error: {json_resp['msg']}")
        if json_resp["data"]["status"] == "success":
            return json_resp
        raise RuntimeError(f"Unexpected task status payload: {json_resp}")

    def convert_visual_prompt(
        self,
        target_image: Union[str, Image.Image],
        prompts: List[Dict],
        return_type: List[str] = ["bbox"],
    ):
        target_image_base64 = encode_image(target_image)
        prompt_images = copy.deepcopy(prompts)

        for prompt in prompt_images:
            prompt["image"] = f"data:image/jpg;base64,{encode_image(prompt['image'])}"

        return {
            "model": "T-Rex-2.0",
            "image": f"data:image/jpg;base64,{target_image_base64}",
            "targets": return_type,
            "prompt": {"type": "visual_images", "visual_images": prompt_images},
        }

    def visual_prompt_inference(
        self,
        target_image: Union[str, Image.Image],
        prompt: List[Dict],
        return_type: List[str] = ["bbox"],
    ):
        payload = self.convert_visual_prompt(target_image, prompt, return_type)
        result = self.call_api(payload)
        try:
            objects = result["data"]["result"]["objects"]
        except Exception as e:
            raise RuntimeError(
                f"T-Rex2 API response missing expected detection objects field. keys={list(result.keys())}"
            ) from e
        detection_result = self.postprocess(objects)
        if "embedding" in return_type:
            base64_embedding = result["data"]["result"]["embedding"]
        else:
            base64_embedding = None
        return detection_result, base64_embedding

    def postprocess(self, object_batches):
        scores = []
        labels = []
        boxes = []
        for obj in object_batches:
            scores.append(obj["score"])
            labels.append(obj["category_id"])
            boxes.append(obj["bbox"])
        return {"scores": scores, "labels": labels, "boxes": boxes}
