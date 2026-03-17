import re

with open('core/network/webgpu_node.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's upgrade the websockets serve call just one more time to make SURE it binds cleanly and ignores ALL origin headers fully
new_text = re.sub(
    r'server = websockets\.serve\(self\._handler, "0\.0\.0\.0", self\.port, process_request=lambda \*args: None\)',
    r'',
    text
)

# And replace it in _run_server
if "Bypass CORS completely" in text:
    pass

