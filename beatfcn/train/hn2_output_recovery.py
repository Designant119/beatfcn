"""Recover stopped-worker text evidence before private-prefix deletion."""
import asyncio
import hashlib
from pathlib import PurePosixPath


async def recover_output(volume,run_id,not_found_error,*,maximum_bytes=32*1024**2):
    from .hn2_safe_runtime import owned_prefixes
    owned_prefixes('/data',run_id)
    prefix='/runs/'+run_id;result={};total=0
    try:entries=await asyncio.wait_for(volume.listdir.aio(prefix,recursive=True),20)
    except (not_found_error,FileNotFoundError):return result
    for entry in entries:
        if getattr(entry.type,'name','') in ('DIRECTORY','DIR'):continue
        path=str(entry.path)
        if not path.startswith('/'):path='/'+path
        if not path.startswith(prefix+'/') or '..' in PurePosixPath(path).parts:
            raise ValueError('Output path escaped owned prefix')
        if PurePosixPath(path).suffix not in ('.json','.jsonl','.log'):
            raise ValueError('Unexpected binary output requires explicit recovery;will not delete')
        async def read():
            nonlocal total
            chunks=[]
            async for chunk in volume.read_file.aio(path):
                total+=len(chunk)
                if total>maximum_bytes:raise ValueError('Recovery exceeds32MiB;will not delete')
                chunks.append(chunk)
            return b''.join(chunks)
        data=await asyncio.wait_for(read(),20)
        result[str(PurePosixPath(path).relative_to(prefix))]=dict(text=data.decode(),sha256=hashlib.sha256(data).hexdigest())
    return result
