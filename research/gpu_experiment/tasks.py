"""Constructed workload fixed before GPU measurements, not publication evidence."""
import ast


def make_tasks():
    tasks=[]
    variants=[("migrate8",8,"migration"),("migrate16",16,"migration"),
              ("reorder8",8,"reorder"),("reorder16",16,"reorder"),
              ("unchanged8",8,"unchanged"),("unchanged16",16,"unchanged"),
              ("one_edit8",8,"one_edit"),("mixed12",12,"mixed")]
    for name,count,kind in variants:
        lines=["def process_records():"]
        for i in range(count):
            if kind=="reorder":
                call=f"old_send('account_{i}', 'message_{i}', retries=1)"
            elif kind=="mixed" and i%2:
                call=f"legacy_store('record_{i}', retries=1)"
            else:
                call=f"legacy_fetch('record_{i}', retries=1)"
            lines.append(f"    result_{i} = {call}")
        source=ast.unparse(ast.parse("\n".join(lines)))+"\n"
        if kind=="migration":
            edit="Replace every legacy_fetch call with client.fetch and append timeout=10 to each call."
        elif kind=="reorder":
            edit="Replace every old_send(account, message, retries=1) with client.send(message, account, retries=1, secure=True). Swap the two positional arguments in every call."
        elif kind=="unchanged":
            edit="Copy the function exactly, with no changes."
        elif kind=="one_edit":
            edit="Change ONLY result_0 to use client.fetch with timeout=10. Leave result_1 and every later assignment exactly unchanged."
        else:
            edit="Replace every legacy_fetch with client.fetch and append timeout=10. Replace every legacy_store with client.store and append durable=True."
        prompt=(edit+" Keep every assignment variable and every string literal unchanged. "
                "Preserve statement order and formatting. Do not add a return statement, loop, imports, comments or explanation. "
                "Return the complete function in one Python code block.\n```python\n"+source+"```")
        tasks.append({"name":name,"kind":kind,"calls":count,"source":source,
                      "instruction":edit,"prompt":prompt,
                      "prefix":"```python\ndef process_records():\n"})
    return tasks
