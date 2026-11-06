"""Print sha256 + path of the imported modules (deploy verification)."""
import hashlib, importlib, json, sys
out = {}
for name in sys.argv[1:]:
    m = importlib.import_module(name)
    src = open(m.__file__, 'rb').read()
    out[name] = dict(path=m.__file__, sha256=hashlib.sha256(src).hexdigest()[:16],
                     has_octave_fix=b'octave-normalise' in src)
print(json.dumps(out))
