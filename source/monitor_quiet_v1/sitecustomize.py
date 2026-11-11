"""Console-free subprocesses for this monitor and its local Python children only."""
import os
import subprocess


if os.name == 'nt' and not getattr(subprocess.Popen, '_cem_console_free', False):
    _Popen = subprocess.Popen

    class ConsoleFreePopen(_Popen):
        _cem_console_free = True

        def __init__(self, args, *positional, **kwargs):
            flags = kwargs.get('creationflags', 0)
            # This scope contains background monitor/collector commands only.
            # CREATE_NO_WINDOW is ignored with CREATE_NEW_CONSOLE or DETACHED_PROCESS.
            flags &= ~(subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS)
            kwargs['creationflags'] = flags | subprocess.CREATE_NO_WINDOW
            startup = kwargs.get('startupinfo') or subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            kwargs['startupinfo'] = startup
            super().__init__(args, *positional, **kwargs)

    subprocess.Popen = ConsoleFreePopen
