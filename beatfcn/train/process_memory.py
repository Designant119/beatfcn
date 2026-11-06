"""Own-process macOS physical footprint, including compressed/ledger memory.

Layout follows the installed SDK mach/task_info.h TASK_VM_INFO revision1
prefix (4-byte packing). This is distinct from RSS and MPS driver allocation.
"""
import ctypes
import sys


class _TaskVMPrefix(ctypes.Structure):
    _pack_=4
    _fields_=[('virtual_size',ctypes.c_uint64),('region_count',ctypes.c_int32),('page_size',ctypes.c_int32)]+[
        (name,ctypes.c_uint64) for name in ('resident_size','resident_size_peak','device','device_peak','internal','internal_peak',
            'external','external_peak','reusable','reusable_peak','purgeable_volatile_pmap','purgeable_volatile_resident',
            'purgeable_volatile_virtual','compressed','compressed_peak','compressed_lifetime','phys_footprint')]


def physical_footprint():
    if sys.platform!='darwin':raise RuntimeError('This guard is macOS-specific')
    lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    lib.task_info.argtypes=[ctypes.c_uint32,ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint32)]
    lib.task_info.restype=ctypes.c_int32
    info=(ctypes.c_uint32*1024)();count=ctypes.c_uint32(1024)
    task=ctypes.c_uint32.in_dll(lib,'mach_task_self_').value
    code=lib.task_info(task,22,info,ctypes.byref(count))
    if code or count.value*4<ctypes.sizeof(_TaskVMPrefix):raise RuntimeError('TASK_VM_INFO unavailable')
    decoded=_TaskVMPrefix.from_buffer_copy(bytes(info)[:ctypes.sizeof(_TaskVMPrefix)])
    return dict(physical_footprint_bytes=int(decoded.phys_footprint),resident_bytes=int(decoded.resident_size),compressed_bytes=int(decoded.compressed))
