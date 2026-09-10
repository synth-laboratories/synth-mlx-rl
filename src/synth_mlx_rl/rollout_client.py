"""The rollout leg of the on-policy lane, over the hosted container contract.

Local training talks to a task container with exactly the schema the hosted
CISPO runner uses -- `training.rollout.request.v1` in, `training.rollout.
summary.v1` out. The environment is not reimplemented for local runs, because an
environment that only exists locally cannot tell you anything about the hosted
one, and the reward definition is the single thing both lanes must agree on.

The difference is the transport. Hosted goes cloud -> SynthTunnel -> the user's
container and hands it a public HTTPS sampler URL. Local is loopback in both
directions: the container calls back into this service's own OpenAI-compatible
endpoint, so the sampler and the trainer are the same resident model and the
behaviour log-probabilities the container returns are already scored under the
policy version being trained.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

ROLLOUT_REQUEST_SCHEMA_VERSION = "training.rollout.request.v1"
ROLLOUT_SUMMARY_SCHEMA_VERSION = "training.rollout.summary.v1"
ROLLOUT_CAPABILITIES_SCHEMA_VERSION = "training.rollout.capabilities.v1"


class RolloutError(RuntimeError):
    """A stable, secret-free failure at the rollout boundary."""


@dataclass(frozen=True, slots=True)
class RolloutAction:
    """One sampled turn, with the token receipt the update is built from."""

    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    log_probs: tuple[float, ...]

    def validate(self) -> None:
        if not self.prompt_token_ids or not self.token_ids:
            raise RolloutError("rollout_action_token_receipt_empty")
        if len(self.log_probs) != len(self.token_ids):
            raise RolloutError("rollout_action_token_receipt_unaligned")


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    rollout_id: str
    reward: float
    actions: tuple[RolloutAction, ...]
    container_digest: str | None
    capability_hash: str | None


class ContainerRolloutClient:
    """Speaks the hosted rollout contract to a task container."""

    def __init__(
        self,
        *,
        base_url: str,
        task_id: str,
        sampler_url: str,
        sampler_token: str,
        bearer_token: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        connection_mode: str = "close",
        timeout_seconds: float = 300.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._task_id = task_id
        self._sampler_url = sampler_url
        self._sampler_token = sampler_token
        self._bearer = bearer_token
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._connection_mode = connection_mode
        self._timeout = timeout_seconds

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"} if data else {}
        if self._bearer:
            headers["Authorization"] = f"Bearer {self._bearer}"
        request = urllib.request.Request(
            f"{self._base}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            detail = (exc.read() or b"").decode(errors="replace")[:400]
            raise RolloutError(f"rollout_container_http_{exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            # Same name the hosted lane uses, so the two lanes fail alike.
            raise RolloutError("training_rollout_env_unreachable") from exc

    def capabilities(self) -> dict[str, Any]:
        """The container's advertised contract, checked before any rollout."""

        payload = self._request("GET", "/training/capabilities")
        if not isinstance(payload, dict):
            raise RolloutError("rollout_capabilities_invalid")
        if payload.get("schema_version") != ROLLOUT_CAPABILITIES_SCHEMA_VERSION:
            raise RolloutError("rollout_capabilities_schema_unsupported")
        if ROLLOUT_REQUEST_SCHEMA_VERSION not in (payload.get("protocol_versions") or []):
            raise RolloutError("rollout_container_does_not_speak_request_v1")
        if payload.get("task_id") != self._task_id:
            raise RolloutError(
                f"rollout_task_mismatch: container serves {payload.get('task_id')!r}, "
                f"configured for {self._task_id!r}"
            )
        dataset_digest = payload.get("dataset_digest")
        if not (
            isinstance(dataset_digest, str)
            and dataset_digest.startswith("sha256:")
            and len(dataset_digest) == len("sha256:") + 64
            and all(character in "0123456789abcdef" for character in dataset_digest[7:])
        ):
            raise RolloutError("rollout_container_dataset_digest_missing_or_invalid")
        return payload

    def rollout(
        self,
        *,
        job_id: str,
        attempt_id: str,
        policy_version: str,
        rollout_id: str | None = None,
        task_instance_id: str | None = None,
        world_ref: str | None = None,
        temperature: float | None = None,
    ) -> RolloutSummary:
        """One episode at a pinned policy version.

        `task_instance_id` selects which task instance runs. It matters more than
        it looks: without it the container serves its default instance and every
        rollout in the run is the same problem, so the policy memorises one row
        and the reward stops meaning anything. A group shares an instance -- that
        is what makes its advantages group-*relative* -- and different groups get
        different ones.
        """

        identity = rollout_id or f"{attempt_id}-{uuid.uuid4().hex[:12]}"
        body = {
            "schema_version": ROLLOUT_REQUEST_SCHEMA_VERSION,
            "job_id": job_id,
            "attempt_id": attempt_id,
            "rollout_id": identity,
            # The key is the rollout identity: a retried transport attempt must
            # not produce a second episode with a different reward.
            "idempotency_key": identity,
            "policy_version": policy_version,
            "task": {
                "task_id": self._task_id,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature if temperature is None else temperature,
                **({"task_instance_id": task_instance_id} if task_instance_id else {}),
                **({"world_ref": world_ref} if world_ref else {}),
            },
            "sampler": {
                "url": self._sampler_url,
                "bearer_token": self._sampler_token,
                "connection_mode": self._connection_mode,
            },
        }
        payload = self._request("POST", "/training/rollouts", body)
        return self._summary(identity, payload)

    @staticmethod
    def _summary(rollout_id: str, payload: Any) -> RolloutSummary:
        if not isinstance(payload, dict):
            raise RolloutError("rollout_summary_invalid")
        if payload.get("schema_version") != ROLLOUT_SUMMARY_SCHEMA_VERSION:
            raise RolloutError("rollout_summary_schema_unsupported")
        reward_block = payload.get("reward")
        if not isinstance(reward_block, dict) or not isinstance(
            reward_block.get("reward"), (int, float)
        ):
            raise RolloutError("rollout_summary_reward_missing")
        raw_actions = payload.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise RolloutError("rollout_summary_actions_missing")
        actions = []
        for raw in raw_actions:
            if not isinstance(raw, dict):
                raise RolloutError("rollout_summary_action_invalid")
            action = RolloutAction(
                prompt_token_ids=tuple(int(v) for v in raw.get("prompt_token_ids") or []),
                token_ids=tuple(int(v) for v in raw.get("token_ids") or []),
                log_probs=tuple(float(v) for v in raw.get("log_probs") or []),
            )
            action.validate()
            actions.append(action)
        return RolloutSummary(
            rollout_id=str(payload.get("rollout_id") or rollout_id),
            reward=float(reward_block["reward"]),
            actions=tuple(actions),
            container_digest=payload.get("container_digest"),
            capability_hash=payload.get("capability_hash"),
        )
