import httpx
import asyncio
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from backend.core.config import settings, MODELS_DIR

class CloudAIService:
    @staticmethod
    def is_tripo_configured() -> bool:
        return bool(settings.TRIPO_API_KEY and len(settings.TRIPO_API_KEY.strip()) > 5)

    @staticmethod
    def is_meshy_configured() -> bool:
        return bool(settings.MESHY_API_KEY and len(settings.MESHY_API_KEY.strip()) > 5)

    @staticmethod
    async def generate_with_tripo(
        image_path: Path,
        model_id: str,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Integrates with Tripo3D API (Image to 3D) when TRIPO_API_KEY is configured.
        """
        if not CloudAIService.is_tripo_configured():
            raise ValueError("TRIPO_API_KEY is not configured in environment variables.")

        headers = {
            "Authorization": f"Bearer {settings.TRIPO_API_KEY}"
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            # 1. Upload image to Tripo
            with open(image_path, "rb") as f:
                upload_res = await client.post(
                    "https://api.tripo3d.ai/v2/openapi/upload",
                    headers=headers,
                    files={"file": (image_path.name, f, "image/png")}
                )
            if upload_res.status_code != 200:
                raise RuntimeError(f"Tripo upload failed: {upload_res.text}")
            
            image_token = upload_res.json()["data"]["image_token"]

            # 2. Start Image-to-3D Task
            payload = {
                "type": "image_to_model",
                "file": {
                    "type": "png",
                    "file_token": image_token
                }
            }
            if prompt:
                payload["prompt"] = prompt

            task_res = await client.post(
                "https://api.tripo3d.ai/v2/openapi/task",
                headers=headers,
                json=payload
            )
            if task_res.status_code != 200:
                raise RuntimeError(f"Tripo task creation failed: {task_res.text}")

            task_id = task_res.json()["data"]["task_id"]

            # 3. Poll for task completion
            for _ in range(60):
                await asyncio.sleep(3)
                poll_res = await client.get(
                    f"https://api.tripo3d.ai/v2/openapi/task/{task_id}",
                    headers=headers
                )
                poll_data = poll_res.json().get("data", {})
                status = poll_data.get("status")
                if status == "success":
                    glb_url = poll_data.get("output", {}).get("model")
                    # Download GLB
                    glb_res = await client.get(glb_url)
                    dest_glb = MODELS_DIR / f"{model_id}.glb"
                    with open(dest_glb, "wb") as out_f:
                        out_f.write(glb_res.content)
                    return {"glb_path": dest_glb, "task_id": task_id}
                elif status in ["failed", "cancelled"]:
                    raise RuntimeError(f"Tripo generation failed: {poll_data.get('message', 'Unknown error')}")

            raise TimeoutError("Tripo 3D generation timed out")

    @staticmethod
    async def generate_with_tripo_text(
        prompt: str,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Integrates with Tripo3D API for Text-to-3D (Scenario C: Prompt Only) when TRIPO_API_KEY is configured.
        """
        if not CloudAIService.is_tripo_configured():
            raise ValueError("TRIPO_API_KEY is not configured in environment variables.")

        headers = {
            "Authorization": f"Bearer {settings.TRIPO_API_KEY}"
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "type": "text_to_model",
                "prompt": prompt
            }
            task_res = await client.post(
                "https://api.tripo3d.ai/v2/openapi/task",
                headers=headers,
                json=payload
            )
            if task_res.status_code != 200:
                raise RuntimeError(f"Tripo text task creation failed: {task_res.text}")

            task_id = task_res.json()["data"]["task_id"]

            for _ in range(60):
                await asyncio.sleep(3)
                poll_res = await client.get(
                    f"https://api.tripo3d.ai/v2/openapi/task/{task_id}",
                    headers=headers
                )
                poll_data = poll_res.json().get("data", {})
                status = poll_data.get("status")
                if status == "success":
                    glb_url = poll_data.get("output", {}).get("model")
                    glb_res = await client.get(glb_url)
                    dest_glb = MODELS_DIR / f"{model_id}.glb"
                    with open(dest_glb, "wb") as out_f:
                        out_f.write(glb_res.content)
                    return {"glb_path": dest_glb, "task_id": task_id}
                elif status in ["failed", "cancelled"]:
                    raise RuntimeError(f"Tripo text generation failed: {poll_data.get('message', 'Unknown error')}")

            raise TimeoutError("Tripo 3D text generation timed out")

    @staticmethod
    async def generate_with_meshy(
        image_path: Path,
        model_id: str,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Integrates with Meshy API (Image-to-3D) when MESHY_API_KEY is configured.
        """
        if not CloudAIService.is_meshy_configured():
            raise ValueError("MESHY_API_KEY is not configured in environment variables.")

        headers = {
            "Authorization": f"Bearer {settings.MESHY_API_KEY}",
            "Content-Type": "application/json"
        }

        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        ext = image_path.suffix.lstrip(".").lower() or "png"
        image_data_uri = f"data:image/{ext};base64,{b64_data}"

        payload = {
            "image_url": image_data_uri,
            "enable_pbr": True,
            "ai_model": "meshy-4"
        }
        if prompt:
            payload["prompt"] = prompt

        async with httpx.AsyncClient(timeout=120.0) as client:
            task_res = await client.post("https://api.meshy.ai/v1/image-to-3d", headers=headers, json=payload)
            if task_res.status_code not in [200, 202]:
                raise RuntimeError(f"Meshy image-to-3d task creation failed: {task_res.text}")

            task_id = task_res.json()["result"]

            for _ in range(60):
                await asyncio.sleep(3)
                poll_res = await client.get(f"https://api.meshy.ai/v1/image-to-3d/{task_id}", headers=headers)
                poll_data = poll_res.json()
                status = poll_data.get("status")
                if status == "SUCCEEDED":
                    glb_url = poll_data.get("model_urls", {}).get("glb")
                    glb_res = await client.get(glb_url)
                    dest_glb = MODELS_DIR / f"{model_id}.glb"
                    with open(dest_glb, "wb") as out_f:
                        out_f.write(glb_res.content)
                    return {"glb_path": dest_glb, "task_id": task_id}
                elif status in ["FAILED", "EXPIRED"]:
                    raise RuntimeError(f"Meshy generation failed: {poll_data.get('task_error', {}).get('message', 'Unknown error')}")

            raise TimeoutError("Meshy 3D generation timed out")

    @staticmethod
    async def generate_with_meshy_text(
        prompt: str,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Integrates with Meshy API (Text-to-3D) when MESHY_API_KEY is configured.
        """
        if not CloudAIService.is_meshy_configured():
            raise ValueError("MESHY_API_KEY is not configured in environment variables.")

        headers = {
            "Authorization": f"Bearer {settings.MESHY_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "mode": "preview",
            "prompt": prompt,
            "art_style": "realistic",
            "should_remesh": True
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            task_res = await client.post("https://api.meshy.ai/v2/text-to-3d", headers=headers, json=payload)
            if task_res.status_code not in [200, 202]:
                raise RuntimeError(f"Meshy text-to-3d task creation failed: {task_res.text}")

            task_id = task_res.json()["result"]

            for _ in range(60):
                await asyncio.sleep(3)
                poll_res = await client.get(f"https://api.meshy.ai/v2/text-to-3d/{task_id}", headers=headers)
                poll_data = poll_res.json()
                status = poll_data.get("status")
                if status == "SUCCEEDED":
                    glb_url = poll_data.get("model_urls", {}).get("glb")
                    glb_res = await client.get(glb_url)
                    dest_glb = MODELS_DIR / f"{model_id}.glb"
                    with open(dest_glb, "wb") as out_f:
                        out_f.write(glb_res.content)
                    return {"glb_path": dest_glb, "task_id": task_id}
                elif status in ["FAILED", "EXPIRED"]:
                    raise RuntimeError(f"Meshy text generation failed: {poll_data.get('task_error', {}).get('message', 'Unknown error')}")

            raise TimeoutError("Meshy 3D generation timed out")

cloud_ai_service = CloudAIService()
