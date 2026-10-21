"""Controlled network association and discretely observed lifecycle intervals."""

from datetime import datetime


def network_intervals(fault, runtime, candidates):
    mapping = {}
    for record in runtime.get("identity", {}).get("records", []):
        if record.get("container_role") == "regular":
            key = (record.get("namespace"), record.get("pod_name"))
            mapping.setdefault(key, set()).add(
                (record.get("pod_uid"), tuple(sorted(record.get("services", []))))
            )
    result = []
    start, end = fault.get("effective_ms"), fault.get("removed_ms")
    if not isinstance(start, int) or not isinstance(end, int) or start >= end:
        return result
    for observed in fault.get("objects", []):
        if (
            observed.get("kind") != "NetworkChaos"
            or observed.get("spec") != fault["document"]["spec"]
        ):
            raise ValueError("Network object spec mismatch")
        for record in observed.get("status", {}).get("experiment", {}).get("containerRecords", []):
            if record.get("selectorKey") != ".Target" or record.get("phase") != "Injected":
                continue
            applied = []
            for event in record.get("events", []):
                if event.get("operation") == "Apply" and event.get("type") == "Succeeded":
                    applied.append(
                        int(
                            datetime.fromisoformat(
                                event["timestamp"].replace("Z", "+00:00")
                            ).timestamp()
                            * 1000
                        )
                    )
            parts = record.get("id", "").split("/")
            identities = mapping.get(tuple(parts), set())
            if not applied or len(identities) != 1:
                continue
            uid, services = next(iter(identities))
            if not uid or max(applied) > start:
                continue
            for service in services:
                if service in candidates:
                    result.append(
                        dict(
                            service=service,
                            start_ms=start,
                            end_ms=end,
                            pod_uid=uid,
                            object_id=record["id"],
                            source="network_participant_association",
                        )
                    )
    return result


def lifecycle_intervals(events, latest, target, effective, max_gap_ms):
    relationship = latest.get("relationship_observation", {})
    if (
        latest.get("observation_gaps") != []
        or relationship.get("status") != "current"
        or relationship.get("errors") != []
    ):
        return []
    if any(event.get("event_type") == "relationship_refresh_failed" for event in events):
        return []
    snapshots = []
    for event in events:
        if event.get("event_type") != "identity_snapshot":
            continue
        timestamp = event.get("received_ms")
        if not isinstance(timestamp, int):
            raise ValueError("Invalid identity snapshot time")
        pods = {
            record["pod_uid"]
            for record in event["records"]
            if target in record.get("services", [])
            and record.get("container_role") == "regular"
            and record.get("pod_ready") is True
            and record.get("container_ready") is True
            and record.get("phase") == "Running"
        }
        routes = [
            endpoint
            for item in event["endpoint_slices"]
            if item.get("service") == target
            for endpoint in item.get("endpoints", [])
            if endpoint.get("ready") is True
        ]
        snapshots.append((timestamp, bool(pods), bool(routes)))
    if any(second[0] <= first[0] for first, second in zip(snapshots, snapshots[1:])):
        raise ValueError("Unordered identity snapshots")
    result, current = [], None
    ready_anchor = None
    for timestamp, pods, routes in snapshots:
        if timestamp < effective:
            if pods and routes:
                ready_anchor = timestamp
            continue
        zero = not pods and not routes
        if zero and ready_anchor is not None and effective - ready_anchor <= max_gap_ms:
            if current is None or timestamp - current["end_ms"] > max_gap_ms:
                if current is not None:
                    return []
                current = dict(
                    service=target,
                    start_ms=timestamp,
                    end_ms=timestamp,
                    observations=1,
                    source="lifecycle_observation_association",
                )
            else:
                current["end_ms"] = timestamp
                current["observations"] += 1
        elif current is not None:
            result.append(current)
            current = None
            break
    return [interval for interval in result if interval["start_ms"] < interval["end_ms"]]
