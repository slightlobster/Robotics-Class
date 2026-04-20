from __future__ import annotations


def replace_latest_queue_item(qobj, item) -> None:
    try:
        while True:
            qobj.get_nowait()
    except Exception:
        pass
    try:
        qobj.put_nowait(item)
    except Exception:
        pass
