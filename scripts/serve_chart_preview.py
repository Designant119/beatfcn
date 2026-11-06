"""Local-only preview server with byte ranges for WebKit audio seeking."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class PreviewHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range = None
        path = Path(self.translate_path(self.path))
        header = self.headers.get('Range')
        if not header or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
        size = path.stat().st_size
        if not match or not any(match.groups()) or not size:
            self.send_error(416, 'Unsupported byte range')
            return None
        left, right = match.groups()
        start = int(left) if left else max(0, size-int(right))
        end = min(size-1, int(right)) if left and right else size-1
        if start > end or start >= size:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        try:
            file = path.open('rb')
        except OSError:
            self.send_error(404)
            return None
        self.byte_range = (start, end)
        self.send_response(206)
        self.send_header('Content-type', self.guess_type(str(path)))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end-start+1))
        self.end_headers()
        file.seek(start)
        return file

    def end_headers(self):
        self.send_header('Accept-Ranges','bytes')
        super().end_headers()

    def copyfile(self, source, outputfile):
        if self.byte_range is None:
            return super().copyfile(source, outputfile)
        remaining = self.byte_range[1]-self.byte_range[0]+1
        while remaining:
            chunk = source.read(min(65536, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


if __name__=='__main__':
    directory = Path(__file__).resolve().parents[1]/'artifacts/chart-preview-20261025'
    ThreadingHTTPServer(('127.0.0.1', 8769), partial(PreviewHandler, directory=str(directory))).serve_forever()
