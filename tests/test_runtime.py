import json
import os
import subprocess
import sys
import tempfile
import time
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import psutil
from webapp import runtime


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        runtime.reopen()
        self.addCleanup(runtime.reopen)

    def test_timeout_kills_parent_and_grandchild(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'child.json'
            script = "import subprocess,sys,json,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path(sys.argv[1]).write_text(json.dumps(p.pid)); time.sleep(60)"
            with (Path(tmp)/'log').open('w') as log, patch.object(runtime, 'memory_limit', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'límite'):
                    runtime.run([sys.executable, '-c', script, str(path)], cwd=tmp, log=log, timeout=1)
            pid = json.loads(path.read_text())
            self.assertFalse(psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE)
            self.assertFalse(runtime._processes)

    def test_memory_guard_and_recovery(self):
        with tempfile.TemporaryDirectory() as tmp, (Path(tmp)/'log').open('w') as log:
            with patch.object(runtime, 'memory_limit', return_value=100), patch.object(runtime, 'memory_usage', side_effect=[10, 90]):
                with self.assertRaisesRegex(RuntimeError, 'memoria'):
                    runtime.run([sys.executable, '-c', 'import time; time.sleep(60)'], cwd=tmp, log=log, timeout=5)
            self.assertFalse(runtime._processes)
            with patch.object(runtime, 'memory_limit', return_value=None):
                code, _ = runtime.run([sys.executable, '-c', 'print("ok")'], cwd=tmp, log=log, timeout=5)
                self.assertEqual(code, 0)

    def test_shutdown_rejects_new_work(self):
        runtime.shutdown()
        with patch.object(runtime, 'memory_limit', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'cerrando'):
                runtime.run([sys.executable, '-c', 'pass'], cwd='.', log=subprocess.DEVNULL, timeout=5)

    def test_shutdown_stops_active_work(self):
        with tempfile.TemporaryDirectory() as tmp, (Path(tmp)/'log').open('w') as log:
            with patch.object(runtime, 'memory_limit', return_value=None):
                result = []
                thread = threading.Thread(target=lambda: result.append(runtime.run(
                    [sys.executable, '-c', 'import time; time.sleep(60)'], cwd=tmp, log=log, timeout=120)))
                thread.start()
                try:
                    deadline = time.monotonic()+5
                    while not runtime._processes and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue(runtime._processes)
                    runtime.shutdown()
                    thread.join(10)
                    self.assertFalse(thread.is_alive())
                    self.assertNotEqual(result[0][0], 0)
                    self.assertFalse(runtime._processes)
                finally:
                    runtime.shutdown()
                    thread.join(10)

    def test_memory_configuration(self):
        with patch.dict(os.environ, {'WEB_MEMORY_LIMIT_MB':'512'}):
            self.assertEqual(runtime.memory_limit(), 512*1024**2)
        with patch.dict(os.environ, {'WEB_MEMORY_LIMIT_MB':'0'}):
            with self.assertRaises(ValueError):
                runtime.memory_limit()


if __name__ == '__main__':
    unittest.main()
